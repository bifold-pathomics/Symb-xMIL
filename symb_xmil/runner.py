"""Symb-xMIL pipeline.

1. Compute f(X_S) for every bag X and subset S of N.
2. Build λ_q(S) for every query q over the same subsets.
3. Compute r_X(q, f) and select q*(X).
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from .perturbation import all_nonempty_subsets, format_subset_key
from .matching import match_queries


def run_symbolic_xmil(
    scorer,
    dataloader,
    results_dir: str,
    query_gen,
    concepts: Iterable | None = None,
    max_subset_size: int | None = None,
    verbose: bool = True,
    save_mapped: bool = False,
) -> dict:
    """Run Symb-xMIL on one dataset split.

    Args:
        scorer:          PerturbationScorer wrapping the trained model.
        dataloader:      DataLoader with batch_size=1.
        results_dir:     Output directory for the CSVs.
        query_gen:       QueryGenerator defining the queries (k, negation, M).
        concepts:        The features N, e.g. [0, 1, 2]. Required.
        max_subset_size: Maximum |S|. None (as in the paper) uses all subsets.
        verbose:         Print progress.
        save_mapped:     Also save CSVs with feature names instead of IDs.

    Returns:
        {"f_df": f(X_S) per bag and subset, "best_df": q*(X) and all alignment scores}
    """
    if not concepts:
        raise ValueError(
            "No features given. Please provide the features N, "
            "e.g. run_symbolic_xmil(..., concepts=[0, 1, 2])."
        )

    concepts = list(concepts)
    if all(isinstance(c, (int, float)) for c in concepts):
        concepts = sorted(concepts)

    os.makedirs(results_dir, exist_ok=True)

    # Load all bags once instead of re-iterating the DataLoader per subset.
    bags = list(dataloader)
    if verbose:
        print(f"[runner] loaded {len(bags)} bags")

    # Queries are built over the same N; refuse a conflicting definition.
    if query_gen.concepts and _concept_set(query_gen.concepts) != _concept_set(concepts):
        raise ValueError(
            f"QueryGenerator was created with concepts={query_gen.concepts}, but "
            f"run_symbolic_xmil uses concepts={concepts}. Define N only once, "
            f"via run_symbolic_xmil(concepts=...)."
        )
    query_gen.concepts = concepts

    # Step 1: score every (bag, subset) pair.
    subsets = all_nonempty_subsets(concepts, max_subset_size)
    if verbose:
        print(f"[runner] concepts={concepts}  "
              f"max_subset_size={max_subset_size}  total subsets={len(subsets)}")

    slide_ids = _extract_slide_ids(bags)
    f_dict: dict[str, list[float]] = {}

    for subset in tqdm(subsets, desc="subsets", disable=not verbose):
        key = format_subset_key(subset)
        scores = []
        for batch in bags:
            score = scorer.score_bag(batch, subset)
            scores.append(score)
            torch.cuda.empty_cache()
        f_dict[key] = scores

    # Build and save the raw f(X_S) table.
    f_df = pd.DataFrame(f_dict)
    f_df.insert(0, "slide_id", slide_ids)

    f_csv = os.path.join(results_dir, "raw_output_for_subsets.csv")
    f_df.to_csv(f_csv, index=False)
    if verbose:
        print(f"[runner] saved {f_csv}  (shape={f_df.shape})")

    # Step 2: build the query truth matrix.
    query_gen.generate_queries()
    all_queries = query_gen.get_all_queries_flat()
    if verbose:
        print(f"[runner] generated {len(all_queries)} queries "
              f"(max_query_size={query_gen.max_query_size}, negation={query_gen.negation})")

    query_gen.convert_queries(all_queries)

    # The truth matrix columns must match the same subsets used in f_dict.
    query_gen.build_truth_matrix(subsets, sparse=True)

    # Step 3: alignment scores r_X(q, f) between f(X_S) and λ_q(S).
    best_df = match_queries(
        f_df,
        query_gen,
        results_dir=results_dir,
        save_mapped=save_mapped,
    )

    if verbose:
        print("[runner] done.")

    return {
        "f_df": f_df,
        "best_df": best_df,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _concept_set(concepts) -> set:
    """N as a set, with group features as sorted tuples."""
    return {
        tuple(sorted(c)) if isinstance(c, (list, tuple, set, frozenset)) else c
        for c in concepts
    }


def _extract_slide_ids(bags: list[dict]) -> list[str]:
    """Read ``sample_ids["slide_id"]`` from each batch as a string."""
    ids = []
    for batch in bags:
        sid = batch.get("sample_ids", {})
        if isinstance(sid, dict):
            val = sid.get("slide_id", f"bag_{len(ids)}")
        else:
            val = sid
        # val may be a list (batch_size=1 still wraps in list)
        if isinstance(val, list):
            val = val[0]
        # default_collate wraps scalar ids in a tensor
        if isinstance(val, torch.Tensor):
            val = val.flatten()[0].item()
        ids.append(str(val))
    return ids
