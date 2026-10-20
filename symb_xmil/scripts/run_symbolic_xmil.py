import argparse
import json
import os
import sys
import time

import torch

# Make the repository root importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from symb_xmil.loaders import load_xmil
from symb_xmil import QueryGenerator, PerturbationScorer, run_symbolic_xmil


def comma_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_concepts(raw):
    if not raw.strip():
        return None
    return [int(x) for x in comma_list(raw)]


def parse_max_subset_size(raw):
    raw = raw.strip().lower()
    if raw in {"", "none", "all", "full"}:
        return None
    return int(raw)


def get_args():
    parser = argparse.ArgumentParser(description="Run Symb-xMIL on a trained xMIL model.")

    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory of a trained xMIL model (contains args.json and checkpoints).")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--patch-metadata-dirs", type=str, nargs="+", required=True,
                        help="Directories with per-patch semantic annotations.")
    parser.add_argument("--semantic-map", type=str, required=True,
                        help="Semantic map M assigning each patch a concept, e.g. 'camelyon'.")
    parser.add_argument("--test-subsets", type=str, nargs="*", default=[])

    parser.add_argument("--concepts", type=str, required=True,
                        help="Semantic concepts N as comma-separated patch-label ids, e.g. '0,1,2'.")
    parser.add_argument("--max-subset-size", type=str, default="all",
                        help="Maximum number of concepts in a subset S (|S|), or 'all' (as in the paper).")
    parser.add_argument("--max-query-size", type=int, default=3,
                        help="k, the maximum number of concepts per query (<= 3).")
    parser.add_argument("--negation", action="store_true")

    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--explained-class", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-save-mapped", action="store_true")

    parser.add_argument("--drop-duplicates", type=str, default="sample")
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main():
    total_start = time.perf_counter()
    args = get_args()

    model_dir = args.model_dir
    results_dir = args.results_dir

    test_subsets = args.test_subsets
    patch_metadata_dirs = args.patch_metadata_dirs
    semantic_map = args.semantic_map

    concepts = parse_concepts(args.concepts)
    max_subset_size = parse_max_subset_size(args.max_subset_size)
    max_query_size = args.max_query_size

    checkpoint = args.checkpoint
    explained_class = args.explained_class
    device = args.device
    save_mapped = not args.no_save_mapped
    negation = args.negation

    drop_duplicates = args.drop_duplicates
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers

    os.makedirs(results_dir, exist_ok=True)

    config = {
        "model_dir": model_dir,
        "results_dir": results_dir,
        "test_subsets": test_subsets,
        "patch_metadata_dirs": patch_metadata_dirs,
        "semantic_map": semantic_map,
        "concepts": concepts,
        "max_subset_size": max_subset_size,
        "max_query_size": max_query_size,
        "checkpoint": checkpoint,
        "explained_class": explained_class,
        "device": device,
        "save_mapped": save_mapped,
        "negation": negation,
        "drop_duplicates": drop_duplicates,
        "test_batch_size": test_batch_size,
        "num_workers": num_workers,
    }

    print("[symb_xmil] config:")
    print(json.dumps(config, indent=4))

    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    timings = {}

    t0 = time.perf_counter()
    model, classifier, test_loader = load_xmil(
        model_dir=model_dir,
        test_subsets=test_subsets,
        train_subsets=None,
        val_subsets=None,
        patch_metadata_dirs=patch_metadata_dirs,
        semantic_map=semantic_map,
        drop_duplicates=drop_duplicates,
        preload_data=False,
        num_workers=num_workers,
        test_batch_size=test_batch_size,
        checkpoint=checkpoint,
        device=device,
        verbose=True,
    )
    timings["load_xmil_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scorer = PerturbationScorer(
        classifier=classifier,
        explained_class=explained_class,
        device=torch.device(device),
    )

    query_gen = QueryGenerator(
        max_query_size=max_query_size,
        semantic_map=semantic_map,
        negation=negation,
    )
    timings["setup_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = run_symbolic_xmil(
        scorer=scorer,
        dataloader=test_loader,
        results_dir=results_dir,
        query_gen=query_gen,
        concepts=concepts,
        max_subset_size=max_subset_size,
        verbose=True,
        save_mapped=save_mapped,
    )
    timings["run_symbolic_xmil_seconds"] = time.perf_counter() - t0

    timings["total_seconds"] = time.perf_counter() - total_start
    timings["total_minutes"] = timings["total_seconds"] / 60.0

    with open(os.path.join(results_dir, "runtime_summary.json"), "w") as f:
        json.dump(timings, f, indent=4)

    print("[symb_xmil] done")
    print("[symb_xmil] runtime summary:")
    print(json.dumps(timings, indent=4))
    print(results["best_df"].head())


if __name__ == "__main__":
    main()