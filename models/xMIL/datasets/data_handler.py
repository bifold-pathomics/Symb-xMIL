import os
import json

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import ast


class MetadataHandler:

    @staticmethod
    def load_split_metadata(
        split_path,
        metadata_dirs,
        subsets,
        label_cols,
        modalities=None,
        merge_on="case_id",
    ):
        """
        Loads and fuses metadata and split information, filtered by the given subsets.

        :param split_path: (str)
        :param metadata_dirs: (list<str>)
        :param subsets: (list<str>)
        :param label_cols: (list<str>)
        :param modalities: (list<str>)
        :param merge_on: (str)
        :return: (pandas.DataFrame)
        """
        # Read split
        split = pd.read_csv(split_path)
        split["subset"] = split["subset"].astype(
            str
        )  # for compatibility with 'subsets' arg
        split = split[split["subset"].isin(subsets)][[merge_on, "subset"] + label_cols]
        # Read and merge case metadata
        case_metadata = pd.DataFrame()
        for idx, metadata_dir in enumerate(metadata_dirs):
            case_met_ = pd.read_csv(os.path.join(metadata_dir, "case_metadata.csv"))
            case_met_.insert(0, "source_id", idx)
            case_metadata = pd.concat(
                [case_metadata, case_met_], axis=0, ignore_index=True
            )
        split = split.merge(
            case_metadata, how="inner", on=merge_on, suffixes=(None, "_ctrl")
        )
        # Sanity check the given data sources
        if "source_id_ctrl" in split:
            if split["source_id"].equals(split["source_id_ctrl"]):
                split = split.drop("source_id_ctrl", axis=1)
            else:
                raise ValueError(
                    "Given data sources do not match the data sources of the split"
                )
        # Read and merge modality-specific metadata
        if modalities is not None:
            for modality in modalities:
                modality_metadata = pd.DataFrame()
                for idx, metadata_dir in enumerate(metadata_dirs):
                    mod_met_ = pd.read_csv(
                        os.path.join(metadata_dir, f"{modality}_metadata.csv")
                    )
                    modality_metadata = pd.concat(
                        [modality_metadata, mod_met_], axis=0, ignore_index=True
                    )
                split = split.merge(
                    modality_metadata,
                    how="inner",
                    on=merge_on,
                    suffixes=(None, "_ctrl"),
                )
        return split


class SlideDataHandler:

    # ─────────────────────────────────────────────────────────────────────
    # PATCH-LABEL SUPPORT — added for Symbolic xMIL.
    # Maps from patch_label_method → {label_name: integer_class_id}.
    # Centralised here so all callers stay in sync.
    # ─────────────────────────────────────────────────────────────────────
    _LABEL_MAPS = {
        "camelyon": {"Empty": -1, "Normal": 0, "Mix": 1, "Tumor": 2},
        "hnsc_majority_border": {
            "Tumor-Neoplastic": 0,
            "Tumor-Inflammatory": 1,
            "Non-Tumor-Inflammatory": 2,
            "Tumor-Connective": 3,
            "Non-Tumor-Connective": 4,
            "Empty": -1,
        },
        # default cell-type vocabulary (used by 'majority', '90_majority', etc.)
        "default": {
            "Empty": 1,
            "Connective": 2,
            "Inflammatory": 3,
            "Neoplastic": 4,
            "Epithelial": 5,
            "Dead": 6,
        },
    }

    @staticmethod
    def _resolve_label_map(patch_label_method):
        """Return the label_name → int map for *patch_label_method*."""
        if "camelyon" in patch_label_method:
            return SlideDataHandler._LABEL_MAPS["camelyon"]
        if patch_label_method in SlideDataHandler._LABEL_MAPS:
            return SlideDataHandler._LABEL_MAPS[patch_label_method]
        return SlideDataHandler._LABEL_MAPS["default"]

    @staticmethod
    def filter_slide_metadata_to_existing_patches(
        slide_metadata, patch_metadata_dirs, patch_label_method
    ):
        """
        Filter ``slide_metadata`` to only include slides that have a patch-level
        annotation metadata file at
        ``<patch_metadata_dir>/<slide_id>/metadata/df.csv``.

        Added for Symbolic xMIL — not part of the original xMIL code.

        :param slide_metadata: (pd.DataFrame) Must contain 'source_id' and 'slide_id'.
        :param patch_metadata_dirs: (list<str>) Directory path per source_id.
        :param patch_label_method: (str) Selects how annotation files are located.
        :return: (pd.DataFrame) Filtered metadata; only slides whose annotation
            file exists.
        """
        if patch_metadata_dirs is None or len(patch_metadata_dirs) == 0:
            print(
                "[patch-label] patch_metadata_dirs is None or empty; "
                "skipping annotation-existence filter."
            )
            return slide_metadata

        existing_slide_ids = set()
        for _, row in slide_metadata.iterrows():
            source_id = row["source_id"]
            slide_id = row["slide_id"]
            if (
                source_id is None
                or source_id >= len(patch_metadata_dirs)
                or patch_metadata_dirs[source_id] is None
            ):
                continue
            annot_path = os.path.join(
                patch_metadata_dirs[source_id], slide_id, "metadata/df.csv"
            )
            if os.path.exists(annot_path):
                existing_slide_ids.add(slide_id)

        filtered = slide_metadata[
            slide_metadata["slide_id"].isin(existing_slide_ids)
        ].reset_index(drop=True)
        print(
            f"[patch-label] Filtered {len(slide_metadata)} → {len(filtered)} "
            f"slides with available annotation metadata."
        )
        return filtered

    # ─────────────────────────────────────────────────────────────────────
    # Original xMIL method, extended for patch-label support.
    # The added parts are marked inline with `# [patch-label]` comments.
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_patch_metadata(
        slide_metadata,
        patches_dirs,
        patch_filters,
        patch_metadata_dirs=None,        # [patch-label] added
        patch_label_method=None,         # [patch-label] added
    ):
        """
        Loads patch metadata from the patch-level metadata frame for all given slides. Filters the patches according
        to the given filters.

        :param slide_metadata: (pd.DataFrame) DataFrame containing slide ids to load patches for.
            Columns required: 'slide_id' identifying where to find the slide; 'source_id' indexing patches_dirs.
        :param patches_dirs: (list<str>) Directories where the patch metadata are stored.
        :param patch_filters: (dict)
            - 'has_annot' (list) Only use patches with some annotation of one of the given classes.
            - 'exclude_annot': (list) Only use patches with no annotation of the given classes.
        :param patch_metadata_dirs: (list<str> | None) [patch-label] Directories holding the
            external annotation files used to assign patch-level labels.  If None, no labels
            are loaded and ``patch_labels`` will be a list of empty tensors.
        :param patch_label_method: (str | None) [patch-label] Strategy for converting raw
            cell / pixel annotations into a single patch label.  See ``get_patch_labels``.
        :return: 5-tuple of lists per slide:
                (feature_indices, patch_ids, patch_labels, positions_x, positions_y)
            ``patch_labels`` was added for Symbolic xMIL.
        """
        print("Loading patch metadata")
        feature_indices, patch_ids, positions_x, positions_y = [], [], [], []
        patch_labels = []                                 # [patch-label] added

        # [patch-label] Resolve label_map once outside the loop.
        label_map = (
            SlideDataHandler._resolve_label_map(patch_label_method)
            if patch_label_method is not None
            else None
        )

        for idx, row in tqdm(slide_metadata.iterrows(), total=len(slide_metadata)):
            source_id, slide_id = row[["source_id", "slide_id"]]
            slide_patches = pd.read_csv(
                os.path.join(patches_dirs[source_id], slide_id, "metadata/df.csv"),
                index_col=0,
            )
            slide_patches = slide_patches.sort_values("patch_id")
            slide_patches.insert(0, "slide_name", slide_id)
            slide_patches.insert(1, "feature_idx", range(len(slide_patches)))
            slide_patches = SlideDataHandler.filter_patches(
                slide_patches, patch_filters
            )
            feature_indices.append(torch.tensor(slide_patches["feature_idx"].values))
            patch_ids.append(torch.tensor(slide_patches["patch_id"].values))
            positions_x.append(
                torch.tensor(
                    slide_patches["position_rel"]
                    .apply(lambda x: ast.literal_eval(x)[0])
                    .values
                )
            )
            positions_y.append(
                torch.tensor(
                    slide_patches["position_rel"]
                    .apply(lambda x: ast.literal_eval(x)[1])
                    .values
                )
            )

            # ─── [patch-label] Load external annotation & map to int labels ──
            if patch_label_method is None or patch_metadata_dirs is None:
                patch_labels.append(torch.empty(0, dtype=torch.long))
                continue

            try:
                annot_df = pd.read_csv(
                    os.path.join(
                        patch_metadata_dirs[source_id],
                        slide_id,
                        "metadata/df.csv",
                    ),
                    index_col=0,
                ).set_index("patch_id")

                aligned = annot_df.reindex(slide_patches["patch_id"])
                raw_labels = SlideDataHandler.get_patch_labels(
                    aligned, patch_label_method
                )
                mapped = raw_labels.map(label_map)
                patch_labels.append(torch.tensor(list(mapped.values), dtype=torch.long))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load patch labels for slide {slide_id}: {e}"
                ) from e

        return feature_indices, patch_ids, patch_labels, positions_x, positions_y

    @staticmethod
    def filter_patches(patch_metadata, patch_filters):
        """
        Filters the patches according to the given filters.

        :param patch_metadata: (pandas.DataFrame) Patch-level metadata, esp. annotation_{idx} columns.
        :param patch_filters: (dict)
            - 'has_annot' (list) Only use patches with some annotation of one of the given classes.
            - 'exclude_annot': (list) Only use patches with no annotation of the given classes.
        :return: (pandas.DataFrame)
        """
        if patch_filters is not None:
            annot_classes = patch_metadata["annotation_classes"].apply(
                lambda x: pd.Series(json.loads(x))
            )
            if "has_annot" in patch_filters:
                annot_cols = [
                    str(annot_cls) for annot_cls in patch_filters["has_annot"]
                ]
                patch_metadata = patch_metadata[
                    annot_classes[annot_cols].sum(axis=1) > 0
                ]
            if "exclude_annot" in patch_filters:
                annot_cols = [
                    str(annot_cls) for annot_cls in patch_filters["exclude_annot"]
                ]
                patch_metadata = patch_metadata[
                    annot_classes[annot_cols].sum(axis=1) == 0
                ]
        return patch_metadata

    @staticmethod
    def load_features(path, feature_indices=None, patch_ids=None):
        """
        Loads features of one slide into RAM. Pass feature_indices or patch ids to only load selected features.

        :param path: (str) Directory where the patch features are stored.
        :param feature_indices: (torch.Tensor) Selected features indices to load (for features saved in list-style).
        :param patch_ids: (torch.Tensor) Selected patch IDs to load (for features saved in dict-style).
        :return: (torch.Tensor) The feature vectors of shape (num_patches, num_features).
        """
        if os.path.exists(f"{path}.npz"):
            features = dict(np.load(f"{path}.npz"))
            keys, features = list(zip(*list(features.items())))
            keys = torch.tensor(list(map(int, list(keys))))
            keys_argsort = torch.argsort(keys)
            features = torch.from_numpy(np.stack(features)).squeeze()
            features = features[keys_argsort]
            if patch_ids is not None:
                patch_ids_to_idx = {
                    patch_id: idx
                    for patch_id, idx in zip(keys.tolist(), keys_argsort.tolist())
                }
                vec_idx = list(map(patch_ids_to_idx.get, patch_ids.tolist()))
                features = features[vec_idx]
        elif os.path.exists(f"{path}.pt"):  # TODO implement sorting by json file
            features = torch.load(f"{path}.pt").squeeze()
            if feature_indices is not None:
                features = features[feature_indices]
        else:
            feature_idx_list = sorted(
                [
                    int(os.path.splitext(file)[0])
                    for file in os.listdir(path)
                    if file.endswith(".pt")
                ]
            )
            features = torch.stack(
                [
                    torch.load(os.path.join(path, f"{idx}.pt"))
                    for idx in feature_idx_list
                ]
            ).squeeze()
            if feature_indices is not None:
                features = features[feature_indices]
        return features.float()

    # ─────────────────────────────────────────────────────────────────────
    # Original xMIL method, extended to also sample patch_labels in lock-step.
    # The ``patch_labels`` arg and the two corresponding `if` branches are the
    # only patch-label additions; everything else is upstream behaviour.
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def sample_features(
        features,
        bag_size,
        sort_sampled_patches=False,
        patch_ids=None,
        positions_x=None,
        positions_y=None,
        patch_labels=None,           # [patch-label] added
    ):
        """
        Randomly samples a number of features from the given feature tensor. Pass patch_ids to recover the identity of
        the sampled features.

        :param features: (torch.Tensor) Feature vectors of shape (num_patches, num_features).
        :param bag_size: (int) Number of features to sample. If this is smaller than the number of feature vectors,
            zero padding will be applied.
        :param patch_ids: (torch.Tensor) List of patch IDs corresponding to the feature vectors.
        :param patch_labels: (torch.Tensor | None) [patch-label] Patch-level integer
            class labels parallel to ``patch_ids``.  Padded with -1.
        :return: 5-tuple (features, patch_ids, positions_x, positions_y, patch_labels).
        """
        idx_list = torch.randperm(features.shape[0])[:bag_size]

        if sort_sampled_patches:
            idx_list, _ = torch.sort(idx_list)

        features = features[idx_list]
        if patch_ids is not None:
            patch_ids = patch_ids[idx_list]
        if positions_x is not None:
            positions_x = positions_x[idx_list]
        if positions_y is not None:
            positions_y = positions_y[idx_list]
        if patch_labels is not None and len(patch_labels) > 0:   # [patch-label]
            patch_labels = patch_labels[idx_list]                # [patch-label]
        if len(features) < bag_size:
            features = torch.cat(
                (features, torch.zeros(bag_size - features.shape[0], features.shape[1]))
            )
            if patch_ids is not None:
                patch_ids = torch.cat(
                    (patch_ids, torch.full((bag_size - len(patch_ids),), -1))
                )
            if positions_x is not None:
                positions_x = torch.cat(
                    (positions_x, torch.full((bag_size - len(positions_x),), -1))
                )
            if positions_y is not None:
                positions_y = torch.cat(
                    (positions_y, torch.full((bag_size - len(positions_y),), -1))
                )
            if patch_labels is not None and len(patch_labels) > 0:           # [patch-label]
                patch_labels = torch.cat(                                    # [patch-label]
                    (patch_labels, torch.full((bag_size - len(patch_labels),), -1))
                )
        return features, patch_ids, positions_x, positions_y, patch_labels

    # ─────────────────────────────────────────────────────────────────────
    # PATCH-LABEL SUPPORT — added for Symbolic xMIL.
    # Converts per-patch annotations (typically a JSON dict of cell-type
    # counts, or a pred_class column from an upstream patch classifier) into
    # a single label string per patch.  Mapping to integers happens later
    # in ``load_patch_metadata`` via ``_LABEL_MAPS``.
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_annotation(x):
        """Safely parse a JSON-encoded annotation cell. Returns None on failure
        or for empty / all-zero dicts (caller should treat as 'Empty')."""
        if pd.isna(x):
            return None
        try:
            ann = json.loads(x)
        except (TypeError, ValueError):
            return None
        if not isinstance(ann, dict) or not ann:
            return None
        if all(v == 0 for v in ann.values()):
            return None
        return ann

    @staticmethod
    def get_patch_labels(patch_metadata, patch_label_method):
        """Compute a per-patch label string from raw annotation columns.

        Added for Symbolic xMIL — not part of the original xMIL code.

        Supported methods:
            - 'majority'                 — argmax of cell-type counts.
            - 'neoplastic_and_majority'  — 'Neoplastic' if any present, else majority.
            - '90_majority'              — majority label only if ≥90% dominant.
            - 'neoplastic_90_majority'   — combines the two above.
            - 'camelyon'                 — tumor-ratio derived from
                                           annotation_classes['2'].
            - 'camelyon_cellvit'         — Tumor/Border/Normal from Neoplastic
                                           ratio in annotation_cellvit.
            - 'hnsc_majority_border'    — majority cell type × border_score.

        :param patch_metadata: (pd.DataFrame) Aligned patch-level annotation.
        :param patch_label_method: (str) One of the strings above.
        :return: (pd.Series) String labels per patch.
        """

        def _majority(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            return max(ann.items(), key=lambda kv: kv[1])[0]

        def _neoplastic_priority(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            if ann.get("Neoplastic", 0) > 0:
                return "Neoplastic"
            return max(ann.items(), key=lambda kv: kv[1])[0]

        def _90_majority(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            total = sum(ann.values())
            label, count = max(ann.items(), key=lambda kv: kv[1])
            return label if count / total >= 0.9 else "Empty"

        def _neoplastic_90_majority(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            if ann.get("Neoplastic", 0) > 0:
                return "Neoplastic"
            total = sum(ann.values())
            label, count = max(ann.items(), key=lambda kv: kv[1])
            return label if count / total >= 0.9 else "Empty"

        def _camelyon(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            try:
                tumor_ratio = float(ann.get("2", 0.0))
            except (TypeError, ValueError):
                return "Empty"
            if tumor_ratio == 1.0:
                return "Tumor"
            if tumor_ratio == 0.0:
                return "Normal"
            if 0.0 < tumor_ratio < 1.0:
                return "Mix"
            return "Empty"

        def _camelyon_cellvit(x):
            ann = SlideDataHandler._parse_annotation(x)
            if ann is None:
                return "Empty"
            total = sum(ann.values())
            ratio = ann.get("Neoplastic", 0) / total
            if ratio == 1.0:
                return "Tumor"
            if 0 < ratio < 1.0:
                return "Border"
            return "Normal"

        def _hnsc_majority_border(row):
            patch_label = _majority(row["cellvit_annotation_noise_3_types_80"])
            border = row.get("border_score")
            if pd.isna(border):
                return "Empty"
            if patch_label == "Neoplastic" and border >= 1:
                return "Tumor-Neoplastic"
            if patch_label == "Inflammatory" and border >= 1:
                return "Tumor-Inflammatory"
            if patch_label == "Inflammatory" and border < 1:
                return "Non-Tumor-Inflammatory"
            if patch_label == "Connective" and border < 1:
                return "Non-Tumor-Connective"
            if patch_label == "Connective" and border >= 1:
                return "Tumor-Connective"
            return "Empty"

        method_dispatch = {
            "majority":                ("annotation_cellvit", _majority),
            "neoplastic_and_majority": ("annotation_cellvit", _neoplastic_priority),
            "90_majority":             ("annotation_cellvit", _90_majority),
            "neoplastic_90_majority":  ("annotation_cellvit", _neoplastic_90_majority),
            "camelyon":                ("annotation_classes", _camelyon),
            "camelyon_cellvit":        ("annotation_cellvit", _camelyon_cellvit),
        }

        if patch_label_method == "hnsc_majority_border":
            return patch_metadata.apply(_hnsc_majority_border, axis=1)
        if patch_label_method in method_dispatch:
            col, fn = method_dispatch[patch_label_method]
            return patch_metadata[col].apply(fn)

        raise ValueError(
            f"Unsupported patch_label_method: {patch_label_method!r}. "
            f"Use one of: {sorted(list(method_dispatch) + ['hnsc_majority_border'])}"
        )


class OmicsDataHandler:
    """Compatibility placeholder for omics-only xMIL datasets.

    The current Symbolic xMIL workspace carries the image/MIL data-loading path.
    Importing ``datasets.mil`` still references ``OmicsDataHandler`` for
    backwards compatibility, so this placeholder keeps image-only imports
    working and fails clearly if omics-only loading is requested.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OmicsDataHandler is not available in this Symbolic xMIL workspace."
        )

class SurvivalDiscretizer:
    """Discretize the survival values based on the quartiles."""

    def __init__(self, survival_col_name="survival_continuous"):
        """
        Initialize SurvivalDiscretizer
        :param survival_col_name: the name of the column containing the survival values.
        """
        self.survival_col = survival_col_name

    def compute_apply_discretizer(self, dataframe):
        """
        Compute the discretizing bins and also apply it to the given data frame.

        :param dataframe: (DataFrame) the dataframe with the survival values in self.survival_col.
        :return:
        """
        # todo: we can add q (# of bins) as the input to have more flexibility.
        dataframe["survival_group"], discretizer_bins = pd.qcut(
            dataframe[self.survival_col], q=4, labels=[0, 1, 2, 3], retbins=True
        )

        # in order to make sure that the least survival is 0 and there is no maximum for it:
        discretizer_bins[0] = 0
        discretizer_bins[-1] = np.inf

        return dataframe, discretizer_bins

    def apply_discretizer(self, dataframe, bins):
        """
        Use the given bins for discretizing the survival values.
        :param dataframe: (DataFrame) the dataframe with the survival values in self.survival_col
        :param bins: (numpy array) the array containing teh bins for the descritization task.
        :return:
        """
        dataframe["survival_group"] = pd.cut(
            dataframe[self.survival_col],
            bins=bins,
            labels=[0, 1, 2, 3],
            include_lowest=True,
        )
        return dataframe
