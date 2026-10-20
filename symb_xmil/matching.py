"""Select the best-aligned query q*(X) for each bag."""

from __future__ import annotations

import os

import pandas as pd


def match_queries(
    f_df: pd.DataFrame,
    query_gen,
    results_dir: str | None = None,
    save_mapped: bool = False,
) -> pd.DataFrame:
    """Return q*(X) per bag, its alignment score, and the scores of all queries.

    Args:
        f_df:        f(X_S) per bag (rows, ``slide_id``) and subset S (columns).
        query_gen:   QueryGenerator with the truth matrix already built.
        results_dir: If given, write the CSVs here.
        save_mapped: Also save alignment scores with concept names instead of IDs.
    """
    scores_df = query_gen.alignment_scores(
        f_df,
        output_dir=results_dir,
        save_mapped_csv=save_mapped,
    )

    best_query = scores_df.idxmax(axis=1)
    best_score = scores_df.max(axis=1)

    summary = scores_df.copy()
    summary.insert(0, "best_alignment_score", best_score)
    summary.insert(0, "best_query", best_query)
    summary.index.name = "slide_id"

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "best_queries.csv")
        summary.to_csv(path)
        print(f"[matching] saved {path}  (shape={summary.shape})")

    return summary
