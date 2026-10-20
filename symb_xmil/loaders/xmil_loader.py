"""Load a trained xMIL model and its test DataLoader from a checkpoint directory.

The repository-local xMIL code in ``models/xMIL`` is added to ``sys.path`` at runtime.
"""

import os
import json
import importlib
import sys
import types
from pathlib import Path
import torch


def _default_xmil_root() -> Path:
    """Return the repository-local xMIL root."""
    return Path(__file__).resolve().parents[2] / "models" / "xMIL"


def _prepend_xmil_path(xmil_root: str | os.PathLike | None = None) -> Path:
    """Put the xMIL root first on sys.path and return it."""
    root = Path(xmil_root) if xmil_root is not None else _default_xmil_root()
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"xMIL root not found: {root}")

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    _drop_non_xmil_modules(root)
    return root


def _drop_non_xmil_modules(xmil_root: Path) -> None:
    """Remove cached top-level modules that would shadow models/xMIL imports."""
    package_names = ("datasets", "models", "xai")
    root_str = str(xmil_root)
    to_drop = []
    for name, module in list(sys.modules.items()):
        is_xmil_package = name in package_names or name.startswith(
            tuple(f"{package}." for package in package_names)
        )
        if not is_xmil_package:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or not str(Path(module_file).resolve()).startswith(root_str):
            to_drop.append(name)

    for name in to_drop:
        sys.modules.pop(name, None)


def _install_xmil_import_compat() -> None:
    """Stub optional xMIL imports that the image-only models never use."""

    if "captum.attr" not in sys.modules:
        captum = sys.modules.get("captum", types.ModuleType("captum"))
        captum_attr = types.ModuleType("captum.attr")

        class IntegratedGradients:  # noqa: D401 - compatibility placeholder
            """Placeholder for optional Captum IntegratedGradients support."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Captum is not installed. Install captum to use IntegratedGradients."
                )

        captum_attr.IntegratedGradients = IntegratedGradients
        captum.attr = captum_attr
        sys.modules["captum"] = captum
        sys.modules["captum.attr"] = captum_attr

    if "models.additive_mil" not in sys.modules:
        additive_mil = types.ModuleType("models.additive_mil")

        class DefaultMILGraph:  # noqa: D401 - compatibility placeholder
            """Placeholder for optional additive MIL support."""

        class xAdditiveMIL:  # noqa: D401 - compatibility placeholder
            """Placeholder for optional additive MIL explanations."""

        def get_additive_mil_model(*args, **kwargs):
            raise ImportError(
                "The optional additive_mil module is not available in models/xMIL."
            )

        additive_mil.DefaultMILGraph = DefaultMILGraph
        additive_mil.xAdditiveMIL = xAdditiveMIL
        additive_mil.get_additive_mil_model = get_additive_mil_model
        sys.modules["models.additive_mil"] = additive_mil

    if "models.mamba_mil" not in sys.modules:
        mamba_mil = types.ModuleType("models.mamba_mil")

        class MambaMILModel:  # noqa: D401 - compatibility placeholder
            """Placeholder for optional MambaMIL support."""

        class xMambaMIL:  # noqa: D401 - compatibility placeholder
            """Placeholder for optional MambaMIL explanations."""

        mamba_mil.MambaMILModel = MambaMILModel
        mamba_mil.xMambaMIL = xMambaMIL
        sys.modules["models.mamba_mil"] = mamba_mil


def _load_xmil_modules(xmil_root: str | os.PathLike | None = None):
    """Load DatasetFactory and ModelFactory from models/xMIL."""
    root = _prepend_xmil_path(xmil_root)
    _install_xmil_import_compat()
    datasets_module = importlib.import_module("datasets")
    models_module = importlib.import_module("models")
    DatasetFactory = datasets_module.DatasetFactory
    ModelFactory = models_module.ModelFactory
    return DatasetFactory, ModelFactory


def _load_checkpoint_into_model(model, checkpoint_path: str, device: torch.device):
    """Load an xMIL checkpoint into a model without importing training.Callback."""
    checkpoint_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_state_dict = checkpoint_dict["model_state_dict"]
    model_state_dict = model.state_dict()

    use_ppeg = getattr(model, "use_ppeg", False)
    compatible_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if "pos_layer" in key and not use_ppeg:
            continue
        if key in model_state_dict:
            compatible_state_dict[key] = value

    model.load_state_dict(compatible_state_dict)
    return model


def load_xmil(
    model_dir: str,
    test_subsets=None,
    patch_metadata_dirs=None,
    semantic_map: str | None = None,
    test_batch_size: int = 1,
    checkpoint: str = "best",
    device: str | torch.device = "cpu",
    verbose: bool = True,
    xmil_root: str | os.PathLike | None = None,
    **dataset_kwargs,
):
    """Load a trained xMIL model and its test DataLoader.

    Args:
        model_dir:           Checkpoint directory with ``args.json`` and ``{best,last}_model.pt``.
        test_subsets:        Test split names, e.g. ["test"]. Defaults to args.json.
        patch_metadata_dirs: Directories with per-patch semantic annotations.
        semantic_map:        Name of M, e.g. "camelyon" (xMIL ``patch_label_method``).
        test_batch_size:     DataLoader batch size.
        checkpoint:          "best" or "last".
        device:              e.g. "cpu" or "cuda".
        verbose:             Print progress.
        xmil_root:           xMIL root; defaults to ``models/xMIL``.
        **dataset_kwargs:    Extra arguments for ``DatasetFactory.build``.

    Returns:
        (model, classifier, test_loader)
    """
    device = torch.device(device) if isinstance(device, str) else device

    # ── Load args from model directory ─────────────────────────────────
    args_path = os.path.join(model_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}")

    with open(args_path) as f:
        model_args = json.load(f)

    if verbose:
        print(f"[xmil_loader] loaded args from {args_path}")
    # Backward compatibility for older xMIL checkpoints.
    if "loss_type" not in model_args:
        if "objective" in model_args:
            model_args["loss_type"] = model_args["objective"]
        else:
            model_args["loss_type"] = "cross-entropy"

    if "optimizer" not in model_args:
        model_args["optimizer"] = "SGD"

    if "grad_clip" not in model_args:
        model_args["grad_clip"] = model_args.get("gradient_clip", None)
    # ── Handle backward compatibility ──────────────────────────────────
    if "num_features" not in model_args:
        model_args["num_features"] = model_args.get("features_dim", 256)
    if "num_classes" not in model_args:
        model_args["num_classes"] = model_args.get("head_dim", model_args.get("n_out", 2))

    if "mode" not in model_args:
        model_args["mode"] = "image-only"

    head_type = model_args.get("head_type", "classification")
    if head_type != "classification":
        raise ValueError(
            f"Only classification models are supported, got head_type={head_type!r}."
        )

    # ── Build dataset args ────────────────────────────────────────────
    dataset_args = dict(model_args)  # start with model args

    # Override with explicit test_subsets if provided
    if test_subsets is not None:
        dataset_args["test_subsets"] = test_subsets

    if patch_metadata_dirs is not None:
        dataset_args["patch_metadata_dirs"] = patch_metadata_dirs
    if semantic_map is not None:
        dataset_args["patch_label_method"] = semantic_map

    # Apply any additional dataset kwargs
    dataset_args.update(dataset_kwargs)

    # Set batch size
    dataset_args["val_batch_size"] = test_batch_size

    # Ensure required fields
    dataset_args["label_cols"] = model_args.get("targets", ["label"])
    dataset_args["targets"] = model_args.get("targets", ["label"])
    dataset_args["head_type"] = head_type

    if verbose:
        print(f"[xmil_loader] dataset config: mode={model_args['mode']}, "
              f"head_type={head_type}")

    # ── Import xMIL modules from repository-local models/xMIL ──────────
    DatasetFactory, ModelFactory = _load_xmil_modules(xmil_root)

    # ── Build dataset and DataLoader ───────────────────────────────────
    _, _, _, _, test_dataset, test_loader = DatasetFactory.build(dataset_args, model_args)

    if test_loader is None:
        raise ValueError(
            f"test_loader is None — check test_subsets: {dataset_args.get('test_subsets')}"
        )

    if verbose:
        print(f"[xmil_loader] test dataset size: {len(test_dataset) if test_dataset else 'N/A'}")

    # ── Build model and classifier ────────────────────────────────────
    model, classifier = ModelFactory.build(model_args, device)

    if verbose:
        print(f"[xmil_loader] model architecture: {model_args.get('aggregation_model', 'unknown')}")

    if checkpoint not in {"best", "last"}:
        raise ValueError(f"checkpoint must be 'best' or 'last', got {checkpoint}")

    ckpt_file = f"{checkpoint}_model.pt"
    ckpt_path = os.path.join(model_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_file} not found in {model_dir}")

    model = _load_checkpoint_into_model(model, ckpt_path, device)

    if verbose:
        print(f"[xmil_loader] loaded checkpoint: {ckpt_file}")
        print(f"[xmil_loader] done — ready for PerturbationScorer")

    return model, classifier, test_loader
