# Symb-xMIL
---
[**Symb-xMIL: Symbolic Explanations for Multiple Instance Learning in Digital Pathology**](https://arxiv.org/abs/2606.06224)

Yanqing Luo, Julius Hense, Niklas Prenissl, Andreas Mock, Klaus-Robert Mueller, Thomas Schnake, Mina Jamshidi Idaji

* accpeted at NeurIPs 2026 as a poster

![main figure](main_figure.png)

**Summary**: Symb-xMIL is a post-hoc symbolic explanation method for multiple instance learning (MIL) models. It perturbs a whole-slide image bag at the semantic concept level, measures the model response for different semantic concept subsets, and matches those responses to pre-defined symbolic queries. The final output supports a per-bag symbolic explanation such as `TUM`, `(TUM and LYM)`, or `(TUM or LYM)`, and a dataset-level analysis based on symbolic representation space.

## Repository Structure

- `symb_xmil/`: symbolic xMIL pipeline, query generation, perturbation scoring, xMIL loader, and symbolic representation space analysis.
- `models/xMIL/`: following [xMIL](https://github.com/bifold-pathomics/xMIL) implementation used for training, testing, and loading MIL checkpoints. In the experiment, we used TransMIL models.
- `semantic/`: notebooks for deriving patch-level semantic labels.
- `./notebooks`: detailed analysis notebooks, such as recovering logic in synthetic datasets, and discovering sample subgroups in head-and-neck cancer.

## Installation

Create and activate a Python environment, then install the tested dependencies in `requirements.txt`. The dependency file records the tested package versions from the development environment.

```bash
pip install -r requirements.txt
```

Alternatively, build the Apptainer/Singularity container from the provided recipe:

```bash
apptainer build symb_xmil.sif symb_xmil.def
```

Semantic labeling dependencies such as [CellViT++](https://github.com/tio-ikim/cellvit-plus-plus) should be installed by following the instructions in their repositories.

## Data Usage

Camelyon16 data, pixel-level annotations, and tumor size information can be downloaded from the [Camelyon16 Grand Challenge](https://camelyon16.grand-challenge.org/Data/). TCGA-HNSCC data can be obtained from [TCGA](https://www.cancer.gov/tcga). HPV status for the HNSCC dataset was downloaded from [cBioPortal](https://www.cbioportal.org/).

## Example Usage

### Semantic labeling

Patch-level semantic labels are required before running symb-xMIL explanation. The `semantic/` folder documents how patch labels are derived for each dataset, including Camelyon labeling and CellViT++-based labeling for the TCGA-HNSCC dataset.

#### Defining your own semantic concepts

The semantic map $\mathcal{M}$ is selected by name (`semantic_map=...`) and is defined in [models/xMIL/datasets/data_handler.py](models/xMIL/datasets/data_handler.py). To use your own concepts on a new dataset:

1. **Provide patch annotations.** For each slide, write `<patch_metadata_dir>/<slide_id>/metadata/df.csv` with a `patch_id` column and a column holding your raw annotation, saved with `DataFrame.to_csv()` so that the first column is the index. Slides without this file are skipped.
2. **Map annotations to concept names.** In `SlideDataHandler.get_patch_labels`, add a function that turns one annotation value into a concept name (return `"Empty"` for unusable patches) and register it in `method_dispatch` as `"<your_map>": ("<annotation_column>", <your_function>)`.
3. **Map concept names to integer IDs.** Add `"<your_map>": {"<ConceptA>": 0, "<ConceptB>": 1, ..., "Empty": -1}` to `SlideDataHandler._LABEL_MAPS`. Patches labeled `-1` are never used as a concept.
4. **Register the names for the queries.** Add the same dictionary to `QueryGenerator.LABEL_MAPS` in [symb_xmil/queries.py](symb_xmil/queries.py) and its key to the lookup list in `QueryGenerator._get_label_map`, so that queries and output files show concept names instead of IDs.

Then pass `semantic_map="<your_map>"` to `load_xmil` and `QueryGenerator`, and the integer IDs of the concepts to explain as `concepts`.

### Model training

Data preprocessing, data splitting, and model training follow the [xMIL](https://github.com/bifold-pathomics/xMIL) workflow. For the datasets evaluated in our paper, we report the following test performance.

| Dataset | Task |Model| Metric |
| --- | --- | --- | --- |
| Synthetic AND | Binary classification | TransMIL| AUROC 0.999 |
| Synthetic OR | Binary classification | TransMIL| AUROC 0.999 |
| Synthetic IMP | Binary classification | TransMIL| AUROC 0.999 |
| Camelyon16 | Binary tumor classification | TransMIL| AUROC 0.995 (0.010) |
| TCGA-HNSCC | Binary HPV status classification | TransMIL| AUROC 0.899 (0.039) |

### Symb-xMIL explanation

#### 1. Load A Trained xMIL Model

```python
import torch

from symb_xmil import QueryGenerator, PerturbationScorer, load_xmil, run_symbolic_xmil

device = "cuda" if torch.cuda.is_available() else "cpu"

model, classifier, test_loader = load_xmil(
    model_dir="/path/to/xmil/checkpoint",
    test_subsets=["test"], # or fold number for heald-out cross validation
    patch_metadata_dirs=["/path/to/patch_label_metadata"],
    semantic_map="camelyon",  # semantic map M; see models/xMIL/datasets/data_handler.py for more
    device=device,
)
```

#### 2. Define The Semantic Concepts for Explanation

The semantic map $\mathcal{M}$ assigns each patch a semantic concept; $\mathcal{N}$ is the set of concepts used for the explanation. For Camelyon16, for example:

```python
# Camelyon16 semantic map:
# {"Empty": -1, "Normal": 0, "Mix": 1, "Tumor": 2}
concepts = [0, 1, 2]
```

The concepts must be given explicitly; `run_symbolic_xmil` raises an error if `concepts` is not provided.

#### 3. Define The Queries

```python
query_gen = QueryGenerator(
    max_query_size=3,         # k: maximum number of concepts per query
    semantic_map="camelyon",  # also supports camelyon_cellvit, hnsc_majority_border
    negation=False,
)
```

The query set $\mathcal{Q}$ consists of symbolic formulas with at most $k$ concepts, such as `Mix` or `(Mix ∨ Tumor)`, with indicator $\lambda_q(\mathcal{S})$ on each subset $\mathcal{S}$. The queries are generated over the concepts $\mathcal{N}$ passed to `run_symbolic_xmil` in step 5.

#### 4. Create The Perturbation Scorer

```python
scorer = PerturbationScorer(
    classifier=classifier,
    explained_class=-1,          # -1 means the last class, often class 1 in binary classification
    device=torch.device(device),
)
```

For one bag $\mathbf{X}$ and semantic concept subset $\mathcal{S}$, the scorer computes $f(\mathbf{X}_{\mathcal{S}})$: the scalar model output on the sub-bag $\mathbf{X}_{\mathcal{S}}$ that keeps only the patches whose semantic value lies in $\mathcal{S}$.

#### 5. Run Symb-xMIL

```python
results = run_symbolic_xmil(
    scorer=scorer,
    dataloader=test_loader,
    results_dir="/path/to/results",
    query_gen=query_gen,
    concepts=concepts,
    max_subset_size=None,  # None: all non-empty subsets S of N, as in the paper
    verbose=True,
    save_mapped=True,
)
```

#### 6. Outputs

Files written to `results_dir`:

- `raw_output_for_subsets.csv`: per-bag sub-bag outputs $f(\mathbf{X}_{\mathcal{S}})$.
- `alignment_scores.csv`: alignment scores $r_{\mathbf{X}}(q, f)$, i.e. Pearson correlations between $f(\mathbf{X}_{\mathcal{S}})$ and each query indicator $\lambda_q(\mathcal{S})$ (`alignment_scores_with_negation.csv` when `negation=True`).
- `best_queries.csv`: best-aligned query $q^*(\mathbf{X})$ and its score per bag (`best_query`, `best_alignment_score`), followed by the alignment scores of all queries.
- `alignment_scores_mapped.csv`: optional copy of the alignment scores with concept IDs mapped to names when `save_mapped=True`.

#### Command line

The same pipeline is available as a script:

```bash
python symb_xmil/scripts/run_symbolic_xmil.py \
    --model-dir /path/to/xmil/checkpoint \
    --results-dir /path/to/results \
    --test-subsets test \
    --patch-metadata-dirs /path/to/patch_label_metadata \
    --semantic-map camelyon \
    --concepts 0,1,2 \
    --max-query-size 3 \
    --device cuda
```

Run with `--help` to see all options. The script additionally writes `run_config.json` and `runtime_summary.json` to the results directory.

#### 7. Symbolic Representation Space

Use `symb_xmil.symbolic_representation_space` to analyze the per-dataset symbolic representation space, where each bag is represented by $\phi(\mathbf{X})$, its vector of alignment scores over all queries.

```python
from symb_xmil.symbolic_representation_space import plot_symbolic_representation_space

plot_df, enriched_df, cluster_report = plot_symbolic_representation_space(
    results["best_df"],  # or "/path/to/results/best_queries.csv"
    min_query_count=3,
    use_kmeans=True,
    output_path="/path/to/results/pca_symbolic_representation_space.png",
)
```

This projects $\phi(\mathbf{X})$ into 2D with PCA, colors bags by $q^*(\mathbf{X})$ (`best_query`), and can optionally add KMeans clusters.

## Contact

For questions or issues, please open a GitHub issue or contact:

[Yanqing Luo](mailto:yanqing.luo@tu-berlin.de)

## Citation

```bibtex
@article{luo2026symb,
  title={Symb-xMIL: Symbolic Explanations for Multiple Instance Learning in Digital Pathology},
  author={Luo, Yanqing and Hense, Julius and Preni{\\ss}l, Niklas and Mock, Andreas and M{\\\"u}ller, Klaus-Robert and Schnake, Thomas and Idaji, Mina Jamshidi},
  journal={arXiv preprint arXiv:2606.06224},
  year={2026}
}
```

## License

This code is provided under the MIT License. Please refer to the license file for details.

Note: the license was updated from CC BY-NC-ND 4.0 in April 2026.
