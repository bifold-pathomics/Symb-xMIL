import ast
import itertools
import os
import re
import pandas as pd


def _sort_concepts(concepts):
    """Plain concepts first, then group concepts."""
    ints = sorted(c for c in concepts if isinstance(c, int))
    groups = sorted((c for c in concepts if isinstance(c, tuple)), key=lambda t: t[0])
    return ints + groups


class QueryGenerator:
    """Generate queries q over N, build λ_q(S), and compute r_X(q, f)."""

    # Semantic maps M: concept name -> patch label (tuple = group concept).
    LABEL_MAPS = {
        'camelyon': {
            'Empty': -1, 'Normal': 0, 'Mix': 1, 'Tumor': 2
        },
        'camelyon_cellvit': {
            'Empty': -1, 'Normal': 0, 'Mix': 1, 'Tumor': 2
        },
        'hnsc_majority_border': {
            'Tumor-Neoplastic': 0, 'Tumor-Inflammatory': 1, 'Non-Tumor-Inflammatory': 2,
            'Tumor-Connective': 3, 'Non-Tumor-Connective': 4, 'Empty': -1
        },
        'toy_experiment': {
            'Rest': (0, 1, 2, 3, 5, 6, 8, 9), '4': 4, '7': 7},

        'default': {
            'Empty': 1, 'Connective': 2, 'Inflammatory': 3,
            'Neoplastic': 4, 'Epithelial': 5, 'Dead': 6
        }
    }

    def __init__(self, concepts=None, max_query_size=3, semantic_map=None, negation=False):
        """
        Args:
            concepts:       N. Leave as None with ``run_symbolic_xmil``, which sets it.
            max_query_size: k (at most 3).
            semantic_map:   Name of M, used to map concept IDs to names.
            negation:       Also generate queries with NOT.
        """
        self.concepts = concepts
        self.max_query_size = max_query_size
        self.semantic_map = semantic_map
        self.negation = negation
        self.label_map = self._get_label_map(semantic_map)
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.queries_by_size = None
        self.df_queries_struct = None
        self.truth_matrix = None

    def _get_label_map(self, semantic_map):
        if semantic_map is None:
            return self.LABEL_MAPS['default']
        for key in ['camelyon_cellvit', 'camelyon', 'hnsc_majority_border', 'toy_experiment']:
            if key in semantic_map:
                return self.LABEL_MAPS[key]
        return self.LABEL_MAPS['default']

    # =========================================================================
    # Query Generation
    # =========================================================================
    
    @staticmethod
    def _concept_str(f) -> str:
        """Concept as used in query strings: 4 -> "4", (0,1,2) -> "(0,1,2)"."""
        if isinstance(f, (set, frozenset)):
            return "(" + ",".join(map(str, sorted(f))) + ")"
        if isinstance(f, (tuple, list)):
            return "(" + ",".join(map(str, sorted(f))) + ")"
        return str(f)

    def generate_queries(self):
        """Generate all queries over N with at most k concepts."""
        if not self.concepts:
            raise ValueError("QueryGenerator.concepts is not set")
        if self.max_query_size > 3:
            raise ValueError("max_query_size <= 3 supported")

        queries_by_size = {1: [], 2: [], 3: []}
        concept_str = self._concept_str

        # One-concept queries.
        if self.max_query_size >= 1:
            for (a,) in itertools.combinations(self.concepts, 1):
                queries_by_size[1].append(concept_str(a))
                if self.negation:
                    queries_by_size[1].append(f"(¬{concept_str(a)})")

        # Two-concept queries.
        if self.max_query_size >= 2:
            for a, b in itertools.combinations(self.concepts, 2):
                sa, sb = concept_str(a), concept_str(b)
                queries_by_size[2].extend([
                    f"({sa} ∧ {sb})",
                    f"({sa} ∨ {sb})",
                ])
                if self.negation:
                    queries_by_size[2].extend([
                        f"(¬{sa} ∧ {sb})",
                        f"({sa} ∧ ¬{sb})",
                        f"(¬{sa} ∨ {sb})",
                        f"({sa} ∨ ¬{sb})",
                        f"(¬{sa} ∨ ¬{sb})",
                    ])

        # Three-concept queries.
        if self.max_query_size >= 3:
            for a, b, c in itertools.combinations(self.concepts, 3):
                sa, sb, sc = concept_str(a), concept_str(b), concept_str(c)
                queries_by_size[3].extend([
                    f"({sa} ∧ {sb} ∧ {sc})",
                    f"({sa} ∨ {sb} ∨ {sc})",
                    f"({sa} ∨ ({sb} ∧ {sc}))",
                    f"({sa} ∧ ({sb} ∨ {sc}))",
                    f"({sb} ∨ ({sa} ∧ {sc}))",
                    f"({sb} ∧ ({sa} ∨ {sc}))",
                    f"({sc} ∨ ({sa} ∧ {sb}))",
                    f"({sc} ∧ ({sa} ∨ {sb}))",
                    f"(({sa} ∧ {sb}) ∨ ({sa} ∧ {sc}) ∨ ({sb} ∧ {sc}))",
                ])
                if self.negation:
                    queries_by_size[3].extend([
                        f"({sa} ∧ {sb} ∧ ¬{sc})",
                        f"({sa} ∧ ¬{sb} ∧ {sc})",
                        f"(¬{sa} ∧ {sb} ∧ {sc})",
                    ])

        self.queries_by_size = queries_by_size
        return queries_by_size

    def get_all_queries_flat(self):
        if self.queries_by_size is None:
            self.generate_queries()
        return [q for size in sorted(self.queries_by_size) for q in self.queries_by_size[size]]

    # =========================================================================
    # Query Parsing
    # =========================================================================

    def parse_query(self, query_str: str) -> dict:
        """Parse a query string into a dict tree with types ATOM / NOT / AND / OR."""
        q = query_str.replace(" ", "")
        return self._parse_expr(q)

    def _parse_expr(self, s: str) -> dict:
        """Parse expression, handling OR as the lowest-precedence operator."""
        # Remove outer parentheses if they wrap the entire expression
        s = self._strip_outer_parens(s)
        
        # Try to split by OR at top level
        parts = self._split_at_top_level(s, '∨')
        if len(parts) > 1:
            return {'type': 'OR', 'children': [self._parse_expr(p) for p in parts]}
        
        # Try to split by AND
        parts = self._split_at_top_level(s, '∧')
        if len(parts) > 1:
            return {'type': 'AND', 'children': [self._parse_expr(p) for p in parts]}
        
        # Handle NOT
        if s.startswith('¬'):
            return {'type': 'NOT', 'child': self._parse_expr(s[1:])}
        
        # Parenthesised string: either a group-concept "(0,1,2,3)" or a
        # sub-expression.  A group concept contains only digits, commas,
        # and spaces — no logic operators.
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            if all(c in '0123456789, ' for c in inner):
                # Group concept -> sorted tuple.
                members = tuple(sorted(int(x.strip()) for x in inner.split(',') if x.strip()))
                return {'type': 'ATOM', 'value': members}
            return self._parse_expr(inner)

        # Group concept stripped of parens, e.g. "0,1,2,3,5,6,8,9"
        if ',' in s and all(c in '0123456789, ' for c in s):
            members = tuple(sorted(int(x.strip()) for x in s.split(',') if x.strip()))
            return {'type': 'ATOM', 'value': members}

        # Plain integer atom
        return {'type': 'ATOM', 'value': int(s)}

    def _strip_outer_parens(self, s: str) -> str:
        """Remove outer parentheses if they match."""
        while s.startswith('(') and s.endswith(')') and self._parens_match(s):
            s = s[1:-1]
        return s

    def _parens_match(self, s: str) -> bool:
        """Check if outer parentheses are matching (not separate groups)."""
        if not s.startswith('(') or not s.endswith(')'):
            return False
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                return False
        return True

    def _split_at_top_level(self, s: str, op: str) -> list:
        """Split string by operator at top level (not inside parentheses)."""
        parts = []
        depth = 0
        current = ""
        i = 0
        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                depth -= 1
                current += c
            elif c == op and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += c
            i += 1
        if current:
            parts.append(current)
        return parts

    # =========================================================================
    # Query Evaluation
    # =========================================================================

    def eval_query(self, query_ast: dict, subset: set) -> int:
        """Return 1 if the parsed query is true on subset S, else 0."""
        qtype = query_ast['type']
        
        if qtype == 'ATOM':
            return int(query_ast['value'] in subset)
        
        if qtype == 'NOT':
            return 1 - self.eval_query(query_ast['child'], subset)
        
        if qtype == 'AND':
            return int(all(self.eval_query(c, subset) for c in query_ast['children']))
        
        if qtype == 'OR':
            return int(any(self.eval_query(c, subset) for c in query_ast['children']))
        
        raise ValueError(f"Unknown query type: {qtype}")

    def eval_query_on_subset(self, query_str: str, subset: tuple) -> int:
        """Convenience method: parse and evaluate in one step."""
        ast = self.parse_query(query_str)
        return self.eval_query(ast, set(subset))

    # =========================================================================
    # Query Conversion (for DataFrame compatibility)
    # =========================================================================

    def convert_query_string(self, q: str) -> dict:
        """Convert a query string to a structured row with metadata."""
        ast = self.parse_query(q)
        concepts = self._extract_concepts(ast)
        negations = self._extract_negated_concepts(ast)
        
        return {
            "query_str": q,
            "query_ast": ast,
            "query_type": self._get_query_type(ast),
            "concepts": tuple(_sort_concepts(concepts)),
            "negations": tuple(_sort_concepts(negations)),
        }

    def _extract_concepts(self, ast: dict) -> set:
        """Extract all semantic concept labels from an AST."""
        if ast['type'] == 'ATOM':
            return {ast['value']}
        if ast['type'] == 'NOT':
            return self._extract_concepts(ast['child'])
        if ast['type'] in ('AND', 'OR'):
            result = set()
            for child in ast['children']:
                result |= self._extract_concepts(child)
            return result
        return set()

    def _extract_negated_concepts(self, ast: dict) -> set:
        """Extract semantic concept labels that are directly negated."""
        if ast['type'] == 'ATOM':
            return set()
        if ast['type'] == 'NOT':
            child = ast['child']
            if child['type'] == 'ATOM':
                return {child['value']}
            return self._extract_negated_concepts(child)
        if ast['type'] in ('AND', 'OR'):
            result = set()
            for child in ast['children']:
                result |= self._extract_negated_concepts(child)
            return result
        return set()

    def _get_query_type(self, ast: dict) -> str:
        """Return a compact structural label for the query."""
        if ast['type'] == 'ATOM':
            return 'ATOM'
        if ast['type'] == 'NOT':
            return 'NOT'
        
        children = ast.get('children', [])
        child_types = [c['type'] for c in children]
        has_nested = any(t in ('AND', 'OR') for t in child_types)
        has_negation = any(t == 'NOT' for t in child_types)
        
        base = ast['type']  # 'AND' or 'OR'
        n_children = len(children)
        
        if n_children == 2 and not has_nested:
            return f"{base}{'_NEG' if has_negation else ''}"
        if n_children == 3 and not has_nested:
            return f"{base}3{'_NEG' if has_negation else ''}"
        if has_nested:
            return f"{base}_NESTED{'_NEG' if has_negation else ''}"
        
        return base

    def convert_queries(self, queries: list) -> pd.DataFrame:
        """Convert query strings to a structured DataFrame."""
        converted = []
        for q in queries:
            struct = self.convert_query_string(str(q))
            struct["query_str_original"] = str(q)
            converted.append(struct)
        self.df_queries_struct = pd.DataFrame(converted)
        return self.df_queries_struct

    # =========================================================================
    # Truth Matrix Building
    # =========================================================================

    @staticmethod
    def _norm_concept(c):
        """Canonical form of one concept: int, or sorted tuple for a group concept."""
        if isinstance(c, (tuple, list)):
            return tuple(sorted(int(x) for x in c))
        if isinstance(c, str):
            c = c.strip()
            if c.isdigit():
                return int(c)
            if c.startswith('(') and c.endswith(')'):
                members = [x.strip() for x in c[1:-1].split(',') if x.strip()]
                return tuple(sorted(int(x) for x in members))
            return int(c)
        return int(c) if isinstance(c, float) else c  # already int

    def normalize_subset(self, subset) -> tuple:
        """Canonical tuple for subset S: plain concepts first, then group concepts."""
        if isinstance(subset, (int, float)):
            return (int(subset),)
        if isinstance(subset, (set, frozenset)):
            # A bare set means one group concept, not several plain concepts.
            return (tuple(sorted(int(x) for x in subset)),)
        if isinstance(subset, str):
            subset = subset.strip()
            if subset.isdigit():
                return (int(subset),)
            return self.normalize_subset(ast.literal_eval(subset))
        if isinstance(subset, (list, tuple)):
            normed = [self._norm_concept(x) for x in subset]
            # Sort: ints first (by value), then tuples (by first element)
            normed.sort(key=lambda x: (1, x[0]) if isinstance(x, tuple) else (0, x))
            return tuple(normed)
        raise ValueError(f"Unsupported subset type: {type(subset)}")

    def subset_to_column_name(self, subset) -> str:
        """Column name of subset S, same format as ``format_subset_key``."""
        normed = self.normalize_subset(subset)

        def _fmt(c):
            if isinstance(c, tuple):
                return "(" + ",".join(map(str, c)) + ")"  # already sorted by _norm_concept
            return str(c)

        if len(normed) == 1:
            return _fmt(normed[0])
        return "(" + ",".join(_fmt(c) for c in normed) + ")"

    def build_truth_vector(self, query_str: str, subset_list: list) -> list:
        """λ_q(S) for one query over a list of subsets."""
        ast = self.parse_query(query_str)
        return [
            self.eval_query(ast, set(self.normalize_subset(subset)))
            for subset in subset_list
        ]

    def build_truth_matrix(self, subset_list: list, sparse: bool = True) -> pd.DataFrame:
        """λ_q(S) for all queries (rows) and subsets (columns)."""
        if self.df_queries_struct is None:
            raise ValueError("Call convert_queries() first")

        normalized = [self.normalize_subset(s) for s in subset_list]
        
        self.df_queries_struct["truth_vector"] = self.df_queries_struct[
            "query_str_original"
        ].apply(lambda q: self.build_truth_vector(q, normalized))

        truth_matrix = pd.DataFrame(
            self.df_queries_struct["truth_vector"].tolist(),
            index=self.df_queries_struct["query_str_original"],
            columns=[self.subset_to_column_name(s) for s in normalized],
        )

        if sparse:
            truth_matrix = truth_matrix.astype(pd.SparseDtype("int", fill_value=0))

        self.truth_matrix = truth_matrix
        return truth_matrix

    # =========================================================================
    # Alignment Scores
    # =========================================================================

    def alignment_scores(self, query_outputs: pd.DataFrame, output_dir: str = None,
                         save_mapped_csv: bool = False,
                         mapped_filename: str = "alignment_scores_mapped.csv") -> pd.DataFrame:
        """r_X(q, f) for every bag X (rows) and query q (columns)."""
        if self.truth_matrix is None:
            raise ValueError("Call build_truth_matrix() first")

        truth = self.truth_matrix
        if hasattr(truth, "sparse"):
            truth = truth.sparse.to_dense()

        outputs = query_outputs.copy()
        if "slide_id" in outputs.columns:
            outputs = outputs.set_index("slide_id")
        outputs.columns = [self.subset_to_column_name(c) for c in outputs.columns]

        common = outputs.columns.intersection(truth.columns)
        if len(common) == 0:
            raise ValueError("No common columns found")

        outputs = outputs[common].apply(pd.to_numeric, errors="coerce")
        truth = truth[common].apply(pd.to_numeric, errors="coerce")

        result = pd.DataFrame(index=outputs.index, columns=truth.index, dtype=float)
        for query in truth.index:
            truth_vec = truth.loc[query]
            result[query] = outputs.apply(lambda row: row.corr(truth_vec), axis=1)

        result.index.name = "slide_id"

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if self.negation:
                result.to_csv(
                    os.path.join(
                        output_dir,
                        "alignment_scores_with_negation.csv",
                    )
                )
            else:
                result.to_csv(os.path.join(output_dir, "alignment_scores.csv"))
            if save_mapped_csv:
                mapped = self.get_mapped_alignment_score_df(result)
                mapped.to_csv(os.path.join(output_dir, mapped_filename))

        return result

    # =========================================================================
    # Label Mapping
    # =========================================================================

    def map_query_numbers_to_labels(self, q: str) -> str:
        """Replace concept IDs in a query string with their names from M."""
        q = str(q)

        # Replace group concepts first: "(0,1,2,3,...)" -> label name.
        def repl_group(m):
            members = tuple(
                sorted(int(x.strip()) for x in m.group(1).split(',') if x.strip())
            )
            return self.inverse_label_map.get(members, m.group(0))

        q = re.sub(r'\(([\d,\s]+)\)', repl_group, q)

        # Replace remaining plain integers → label name
        def repl_int(m):
            val = int(m.group(0))
            return str(self.inverse_label_map.get(val, m.group(0)))

        return re.sub(r'\b\d+\b', repl_int, q)

    def get_mapped_alignment_score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with query columns mapped to labels."""
        mapped = df.copy()
        mapped.columns = [self.map_query_numbers_to_labels(c) for c in mapped.columns]
        return mapped
