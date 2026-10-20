"""Compute f(X_S): the model output on the sub-bag X_S.

Works with any classifier exposing ``validation_step(batch, softmax=False)``
that returns ``(preds, ...)`` with preds of shape [B, num_classes, num_targets].
For other output formats, subclass PerturbationScorer and override ``_reduce``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batch masking  (single-bag, no DataLoader rebuild)
# ---------------------------------------------------------------------------

def apply_mask(batch: dict, subset: list, device: torch.device) -> dict:
    """Return the sub-bag X_S: a copy of ``batch`` keeping only patches with concept in S."""
    label_set = np.array(sorted(_flatten(subset)))
    patch_labels = batch["patch_labels"].cpu().numpy()        # [1, n_patches]
    keep = np.array([lbl in label_set for lbl in patch_labels[0]], dtype=bool)
    keep_t = torch.from_numpy(keep)                           # CPU bool tensor

    masked = {
        k: (v.clone() if isinstance(v, torch.Tensor) else copy.copy(v))
        for k, v in batch.items()
    }

    masked["features"] = batch["features"].cpu()[:, keep_t, :].to(device)

    for key in ("patch_ids", "patch_labels"):
        if key in masked and isinstance(masked[key], torch.Tensor):
            masked[key] = batch[key].cpu()[:, keep_t].to(device)

    if "targets" in masked and isinstance(masked["targets"], torch.Tensor):
        masked["targets"] = batch["targets"].to(device)

    masked["bag_size"] = torch.tensor([int(keep_t.sum())], device=device)
    masked["sample_ids"] = batch["sample_ids"]

    return masked


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PerturbationScorer:
    def __init__(
        self,
        classifier,
        explained_class: int = -1,
        device: torch.device | None = None,
    ):
        """
        Args:
            classifier:      Model with ``validation_step(batch, softmax=False)``.
            explained_class: Class whose output is f(X_S); -1 = last class.
            device:          Device for the sub-bags. Defaults to the classifier's device.
        """
        self.classifier = classifier
        self.explained_class = explained_class
        self.device = device or getattr(classifier, "device", torch.device("cpu"))

    def score_bag(self, batch: dict, subset: list) -> float:
        """Return f(X_S) for one bag and subset S (0.0 if X_S is empty)."""
        masked = apply_mask(batch, subset, self.device)

        if masked["features"].numel() == 0 or masked["features"].shape[1] == 0:
            return 0.0

        with torch.no_grad():
            preds, *_ = self.classifier.validation_step(masked, softmax=False)

        return self._reduce(preds)

    def _reduce(self, preds) -> float:
        """Scalar output of the explained class; override for other models."""
        cls = self.explained_class
        return float(preds[:, cls, 0].detach().cpu().reshape(-1)[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten(x):
    """Flatten nested concepts (e.g. group concepts) to a list of labels."""
    if isinstance(x, (list, tuple, set, frozenset)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    if isinstance(x, torch.Tensor):
        return _flatten(x.detach().cpu().tolist())
    return [x]
