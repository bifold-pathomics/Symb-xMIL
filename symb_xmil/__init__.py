"""Symb-xMIL: symbolic explanations for MIL models.

Notation (as in the paper), used throughout the package:
    X        bag (slide) of patches
    M        semantic map, assigns each patch a feature
    N        set of features
    S        non-empty subset of N
    X_S      sub-bag of patches whose feature is in S
    f(X_S)   model output on X_S
    k        maximum number of features per query
    λ_q(S)   indicator of query q on S
    r_X(q,f) alignment score: Pearson correlation of f(X_S) and λ_q(S) over S
    q*(X)    best-aligned query
"""

from .scorer import PerturbationScorer, apply_mask
from .matching import match_queries
from .runner import run_symbolic_xmil
from .queries import QueryGenerator
from .loaders import load_xmil

__all__ = [
    "PerturbationScorer",
    "apply_mask",
    "match_queries",
    "run_symbolic_xmil",
    "QueryGenerator",
    "load_xmil",
]
