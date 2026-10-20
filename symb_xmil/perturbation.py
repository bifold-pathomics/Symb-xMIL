from itertools import combinations

import numpy as np
import torch


def all_nonempty_subsets(concepts, max_subset_size=None):
    """All non-empty subsets S of N with |S| <= max_subset_size (None: all 2^|N|-1)."""
    concept_list = list(concepts)
    max_size = len(concept_list) if max_subset_size is None else min(max_subset_size, len(concept_list))
    return [
        list(subset)
        for size in range(1, max_size + 1)
        for subset in combinations(concept_list, size)
    ]


def flatten(x):
    """Flatten nested concepts (e.g. group concepts) to a list of labels."""
    if isinstance(x, (list, tuple, set, frozenset)):
        flat = []
        for item in x:
            flat.extend(flatten(item))
        return flat
    if isinstance(x, torch.Tensor):
        return flatten(x.detach().cpu().tolist())
    return [x]


def mask(concept_subset, data_loader, device):
    """Legacy: DataLoader of sub-bags X_S. Superseded by ``scorer.apply_mask``."""
    label_set = flatten(concept_subset)
    concept_array = np.array(sorted(label_set))

    modified_batches = []

    for batch in data_loader:
        modified_batch = {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in batch.items()
        }

        patch_labels = batch["patch_labels"]  # [1, n_patches]
        labels = patch_labels.cpu().numpy()
        keep_mask = np.array([label in concept_array for label in labels[0]], dtype=bool)
        keep_mask = torch.from_numpy(keep_mask)

        for key in ["features", "patch_ids", "patch_labels"]:
            if key in modified_batch and isinstance(modified_batch[key], torch.Tensor):
                if key == "features":
                    features = modified_batch[key].cpu()
                    features = features[:, keep_mask, :]
                    modified_batch[key] = features.to(device)
                else:
                    tensor = modified_batch[key].cpu()
                    modified_batch[key] = tensor[:, keep_mask].to(device)

        if "targets" in modified_batch:
            modified_batch["targets"] = modified_batch["targets"].squeeze(-1).to(device)

        modified_batch["bag_size"] = torch.tensor([keep_mask.sum().item()], device=device)

        if "sample_ids" in modified_batch:
            modified_batch["sample_ids"] = batch["sample_ids"]

        modified_batches.append(modified_batch)

    class MaskedDataset(torch.utils.data.Dataset):
        def __init__(self, batches):
            self.batches = batches

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, idx):
            return self.batches[idx]

    masked_dataset = MaskedDataset(modified_batches)
    return torch.utils.data.DataLoader(
        masked_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: {
            "features": torch.cat([batch["features"] for batch in x], dim=0),
            "targets": torch.cat([batch["targets"] for batch in x], dim=0),
            "sample_ids": {
                col: [batch["sample_ids"][col] for batch in x]
                for col in x[0]["sample_ids"]
            },
            "bag_size": torch.cat([batch["bag_size"] for batch in x], dim=0),
            "patch_ids": torch.cat([batch["patch_ids"] for batch in x], dim=0),
            "patch_labels": torch.cat([batch["patch_labels"] for batch in x], dim=0),
        },
    )


def _concept_to_str(element) -> str:
    """4 -> "4"; group concept (0,1,2) -> "(0,1,2)"."""
    if isinstance(element, (tuple, list)):
        return "(" + ",".join(map(str, sorted(element))) + ")"
    return str(element)


def format_subset_key(subset) -> str:
    """Column name of subset S in the f(X_S) table.

    [4] -> "4", [4, 7] -> "(4,7)", [4, (0,1,2,3)] -> "(4,(0,1,2,3))"
    """
    if isinstance(subset, (set, frozenset)):
        return "(" + ",".join(map(str, sorted(subset))) + ")"

    items = list(subset) if isinstance(subset, (list, tuple)) else [subset]

    if len(items) == 1:
        return _concept_to_str(items[0])

    return "(" + ",".join(_concept_to_str(e) for e in items) + ")"
