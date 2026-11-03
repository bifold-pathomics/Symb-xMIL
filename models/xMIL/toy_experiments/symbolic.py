"""Symb-xMIL helpers for the synthetic MNIST toy experiments.

Requires the repository root on ``sys.path`` so that ``symb_xmil`` is importable.
"""

import torch

from symb_xmil import PerturbationScorer


class ToyPerturbationScorer(PerturbationScorer):
    def score_bag(self, batch: dict, subset: list) -> float:
        labels = batch.get("patch_labels", batch.get("numbers"))
        if labels is None:
            raise KeyError("Toy Symb-xMIL scoring needs `numbers` or `patch_labels` in each batch.")

        labels_cpu = labels.detach().cpu()
        if labels_cpu.ndim == 2:
            labels_1d = labels_cpu[0]
        else:
            labels_1d = labels_cpu

        label_set = torch.tensor(_flatten(subset), dtype=labels_1d.dtype)
        keep = torch.isin(labels_1d, label_set)

        if int(keep.sum()) == 0:
            return 0.0

        masked = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                masked[key] = value.clone()
            else:
                masked[key] = value

        features = batch["features"]
        if features.ndim == 3:
            masked["features"] = features[:, keep, :].to(self.device)
        elif features.ndim == 2:
            masked["features"] = features[keep, :].to(self.device)
        else:
            raise ValueError(f"Unsupported toy feature shape: {tuple(features.shape)}")

        for key in ("numbers", "patch_labels", "patch_ids"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                tensor = batch[key]
                if tensor.ndim == 2:
                    masked[key] = tensor[:, keep].to(self.device)
                elif tensor.ndim == 1:
                    masked[key] = tensor[keep].to(self.device)
                else:
                    masked[key] = tensor.to(self.device)

        if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
            masked["targets"] = batch["targets"].to(self.device)

        masked["bag_size"] = torch.tensor([int(keep.sum())], device=self.device)
        masked["sample_ids"] = batch["sample_ids"]

        with torch.no_grad():
            preds, *_ = self.classifier.validation_step(masked, softmax=True)
        return self._reduce(preds)


def _flatten(x):
    if isinstance(x, (list, tuple, set, frozenset)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    if isinstance(x, torch.Tensor):
        return _flatten(x.detach().cpu().tolist())
    return [x]


def toy_concepts(num_numbers):
    """Features N = [4, 7, {all other digits}]."""
    rest = tuple(i for i in range(num_numbers) if i not in (4, 7))
    return [4, 7, rest]
