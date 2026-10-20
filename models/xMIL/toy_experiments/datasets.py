import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature-cache paths. Each directory holds class_0.pt ... class_9.pt.
# The root defaults to <repo>/data/toy_features and can be changed with the
# SYMB_XMIL_TOY_FEATURES environment variable, or per call via `features_path`.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FEATURE_ROOT = os.environ.get("SYMB_XMIL_TOY_FEATURES", os.path.join(_REPO_ROOT, "data", "toy_features"))

FEATURE_PATHS = {
    name: os.path.join(FEATURE_ROOT, subdir)
    for name, subdir in {
        "mnist_resnet18":         "mnist_resnet18",
        "mnist_resnet18_blur":    "mnist_resnet18_blur",
        "mnist_resnet18_color":   "mnist_resnet18_color",
        "mnist_resnet18_marker":  "mnist_resnet18_marker",
        "cmnist_random_resnet18": "cmnist_random_resnet18",
        "mnist_decoy":            "decoy_mnist_resnet18",
    }.items()
}

# Feature types that use a pre-extracted ResNet18 cache (all share the same
# load+stack pattern; noise is skipped only for the base "mnist_resnet18" type).
_RESNET_CACHED_TYPES = set(FEATURE_PATHS)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_number_mil_dataset(
    dataset_type,
    num_numbers,
    num_bags,
    num_instances,
    features_type="mnist_resnet18",
    sampling="hierarchical",
    noise=1.0,
    threshold=1,
    features_path=None,
):
    """Instantiate an AND / OR / MAT MIL dataset.

    Parameters
    ----------
    dataset_type : {"and", "or", "mat"}
    num_numbers  : int — vocabulary size (number of distinct digit classes).
    num_bags     : int — number of bags in the dataset.
    num_instances: int or (min, max) — fixed or random bag size.
    features_type: str — see FEATURE_PATHS for valid cached types, or "onehot".
    sampling     : {"hierarchical", "pure"}
    noise        : float — std of Gaussian noise added to cached features
                   (ignored for plain "mnist_resnet18").
    threshold    : int — kept for interface compatibility; currently unused.
    features_path: str or None — override FEATURE_PATHS lookup.
    """
    cls_map = {"and": AndDataset, "or": OrDataset, "mat": MatDataset}
    if dataset_type not in cls_map:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Choose from {list(cls_map)}.")
    return cls_map[dataset_type](
        num_numbers=num_numbers,
        num_bags=num_bags,
        num_instances=num_instances,
        features_type=features_type,
        sampling=sampling,
        noise=noise,
        threshold=threshold,
        features_path=features_path,
    )


# ---------------------------------------------------------------------------
# One-time feature extraction utility (not called during training)
# ---------------------------------------------------------------------------

def get_MNIST_features(root, download=True):
    """Extract ResNet18 features from MNIST and return a digit → tensor dict.

    This is a one-time utility; run it once to build the cached .pt files
    that the dataset classes load at training time.
    """
    transform_1 = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True,  download=download, transform=transform_1)
    testset  = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform_1)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(testset,  batch_size=64, shuffle=False)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False).eval()
    model.fc = torch.nn.Identity()
    transform_2 = weights.transforms()

    features, targets = [], []
    for batch in tqdm(train_loader):
        with torch.no_grad():
            features.append(model(transform_2(batch[0])))
            targets.append(batch[1])
    for batch in tqdm(test_loader):
        with torch.no_grad():
            features.append(model(transform_2(batch[0])))
            targets.append(batch[1])

    features = torch.concat(features)
    targets  = torch.concat(targets)
    return {t: features[targets == t] for t in range(10)}


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def bag_collate_fn(batch_list):
    """Custom collate for bags with variable-length patch lists."""
    col_batch = {}
    for key in batch_list[0].keys():
        if key == "features":
            col_batch[key] = torch.concat([batch[key] for batch in batch_list])
        elif key in ("sample_ids",):
            col_batch[key] = {
                col: [batch[key][col] for batch in batch_list]
                for col in batch_list[0][key]
            }
        elif key == "targets":
            col_batch[key] = torch.stack([batch[key] for batch in batch_list])
        elif key == "numbers":
            col_batch[key] = torch.stack([batch[key] for batch in batch_list])
        elif key == "evidence":
            col_batch[key] = default_collate([{key: b[key]} for b in batch_list])[key]
        elif isinstance(batch_list[0][key], dict):
            # Some fields (e.g., 'relevance') are dicts per-sample; keep as list of dicts
            col_batch[key] = [batch[key] for batch in batch_list]
        else:
            col_batch[key] = torch.tensor([batch[key] for batch in batch_list])
    return col_batch


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class NumberMILDataset(Dataset):
    """MIL bag dataset built from MNIST digit features.

    Parameters
    ----------
    num_numbers  : int — vocabulary size (0 … num_numbers-1).
    num_bags     : int — dataset size.
    num_instances: int or (min, max) — bag size or random range.
    noise        : float — std of Gaussian noise added to cached features.
                   Has no effect for the base "mnist_resnet18" type (intentional).
    threshold    : int — kept for interface compat; unused in base class.
    sampling     : {"hierarchical", "pure"}
        - "hierarchical": each digit is independently included/excluded.
        - "pure"        : reject bags that don't contain at least one of
                          digits 4, 7, *and* one other digit.
    features_type: str — "onehot" for identity-matrix features, or any key in
                   FEATURE_PATHS for pre-extracted ResNet18 embeddings.
    features_path: str or None — override the FEATURE_PATHS lookup.
    """

    def __init__(
        self,
        num_numbers,
        num_bags,
        num_instances,
        noise=1.0,
        threshold=1,
        sampling="hierarchical",
        features_type="mnist_resnet18",
        features_path=None,
    ):
        if features_type == "mnist":
            raise ValueError(
                "'mnist' raw-image features are not supported. "
                "Use 'mnist_resnet18' (pre-extracted) or 'onehot' instead."
            )

        self.num_numbers  = num_numbers
        self.num_bags     = num_bags
        self.num_instances = num_instances
        self.thr          = threshold

        # Load feature cache once before the bag loop.
        if features_type in _RESNET_CACHED_TYPES:
            root = features_path or FEATURE_PATHS[features_type]
            data_dict = {
                idx: torch.load(os.path.join(root, f"class_{idx}.pt"))
                for idx in range(10)
            }
        else:
            data_dict = None   # onehot — generated per-patch on the fly

        self.numbers:  list[torch.Tensor] = []
        self.features: list[torch.Tensor] = []

        for bag_idx in range(num_bags):
            # --- bag size ---------------------------------------------------
            if isinstance(num_instances, (tuple, list)):
                current_bag_size = np.random.randint(num_instances[0], num_instances[1] + 1)
            else:
                current_bag_size = num_instances

            # --- number sampling --------------------------------------------
            if sampling == "hierarchical":
                candidates = np.where(np.random.randint(0, 2, size=num_numbers) == 1)[0]
                if len(candidates) == 0:
                    candidates = np.arange(num_numbers)
                sampled = np.random.choice(candidates, size=current_bag_size, replace=True)

            elif sampling == "pure":
                attempts = 0
                max_attempts = 100000
                while True:
                    attempts += 1
                    if attempts > max_attempts:
                        print(f"[WARNING] Bag {bag_idx}: Exceeded {max_attempts} attempts for pure sampling!")
                        print(f"[WARNING] bag_size={current_bag_size}, num_numbers={num_numbers}")
                        print(f"[WARNING] Falling back to hierarchical sampling for this bag")
                        candidates = np.where(np.random.randint(0, 2, size=num_numbers) == 1)[0]
                        if len(candidates) == 0:
                            candidates = np.arange(num_numbers)
                        sampled = np.random.choice(candidates, size=current_bag_size, replace=True)
                        break
                    
                    pool = np.where(np.random.randint(0, 2, size=num_numbers) == 1)[0]
                    if len(pool) == 0:
                        continue
                    sampled = np.random.choice(pool, size=current_bag_size, replace=True)
                    sampled_set = set(sampled)
                    if 4 in sampled_set and 7 in sampled_set and len(sampled_set - {4, 7}) > 0:
                        break
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling!r}.")

            self.numbers.append(sampled)

            # --- feature construction ----------------------------------------
            if features_type == "onehot":
                bag_feats = (
                    np.random.normal(0.0, noise, (current_bag_size, num_numbers))
                    + np.eye(num_numbers)[sampled]
                )
            else:
                # All cached ResNet18 variants share the same load+stack pattern.
                # Noise is added for all augmented variants; base "mnist_resnet18" has no noise.
                bag_feats = np.stack([
                    data_dict[n][np.random.choice(data_dict[n].shape[0])].numpy()
                    for n in sampled
                ])
                if features_type != "mnist_resnet18":
                    bag_feats = bag_feats + np.random.normal(0.0, noise, bag_feats.shape)

            self.features.append(bag_feats)

        self.numbers  = [torch.tensor(b)                             for b in self.numbers]
        self.features = [torch.tensor(f, dtype=torch.float32) for f in self.features]
        self.num_features = self.features[0].shape[-1] if self.features else None
        print(f"Dataset ready: {num_bags} bags, feature_dim={self.num_features}")

    @property
    def num_classes(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        return {
            "features":   self.features[idx],
            "bag_size":   self.numbers[idx].shape[0],
            "numbers":    self.numbers[idx],
            "sample_ids": {"source_id": 0, "slide_id": idx},
        }


# ---------------------------------------------------------------------------
# Concrete datasets — AND / OR / MAT
# ---------------------------------------------------------------------------
# Positive rule:
#   AND  — bag contains both 4 and 7
#   OR   — bag contains 4 or 7 (or both)
#   MAT  — bag contains 4 but NOT 7  →  label 0; everything else → label 1
# ---------------------------------------------------------------------------

_POS_NUMBERS = torch.tensor([4, 7])


class AndDataset(NumberMILDataset):
    """Label = 1 iff both digit 4 and digit 7 appear in the bag."""

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        label = int(count[4] > 0 and count[7] > 0)
        targets = torch.tensor([label])

        pos_mask = torch.isin(item["numbers"], _POS_NUMBERS).long()
        relevance = {0: -pos_mask + (1 - pos_mask), 1: pos_mask - (1 - pos_mask)}
        return {**item, "targets": targets, "relevance": relevance}


class OrDataset(NumberMILDataset):
    """Label = 1 iff digit 4 or digit 7 (or both) appear in the bag."""

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        label = int(count[4] > 0 or count[7] > 0)
        targets = torch.tensor([label])

        pos_mask = torch.isin(item["numbers"], _POS_NUMBERS).long()
        relevance = {0: 1 - pos_mask, 1: pos_mask}
        return {**item, "targets": targets, "relevance": relevance}


class MatDataset(NumberMILDataset):
    """Label = 0 if only digit 4 is present (no 7); label = 1 otherwise."""

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        label = int(not (count[4] > 0 and count[7] == 0))
        targets = torch.tensor([label])

        pos_mask = torch.isin(item["numbers"], _POS_NUMBERS).long()
        relevance = {0: pos_mask, 1: pos_mask}
        return {**item, "targets": targets, "relevance": relevance}
