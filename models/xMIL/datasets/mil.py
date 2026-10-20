import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.data_handler import (
    MetadataHandler,
    SlideDataHandler,
    OmicsDataHandler,
    SurvivalDiscretizer,
)


class MILSlideDataset(Dataset):

    # ─────────────────────────────────────────────────────────────────────
    # Original xMIL ``MILSlideDataset`` extended with patch-label support.
    # All patch-label additions are marked inline with `# [patch-label]`.
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        split_path,
        metadata_dirs,
        subsets,
        patches_dirs,
        features_dirs,
        label_cols,
        bag_size=None,
        sort_sampled_patches=False,
        num_repetitions=1,
        patch_filters=None,
        preload_features=False,
        drop_duplicates="sample",
        max_bag_size=None,
        min_bag_size=0,
        survival=False,
        survival_bins=None,
        reference_target=0,
        patch_metadata_dirs=None,                   # [patch-label] added
        patch_label_method=None,                    # [patch-label] added
    ):
        super(MILSlideDataset, self).__init__()
        # Save args
        self.bag_size = bag_size
        self.num_repetitions = num_repetitions
        self.features_dirs = features_dirs
        self.label_cols = label_cols
        self.num_targets = len(self.label_cols)
        self.sort_sampled_patches = sort_sampled_patches
        self.survival = survival
        self.survival_bins = survival_bins
        self.patch_label_method = patch_label_method  # [patch-label] added

        # Load metadata, slide data, and match them
        print(f"Loading dataset for subsets: {subsets}")
        split_metadata = MetadataHandler.load_split_metadata(
            split_path, metadata_dirs, subsets, label_cols, modalities=["slide"]
        )

        if self.survival:
            survival_discretizer = SurvivalDiscretizer(label_cols[0])

            if self.survival_bins is None:
                split_metadata, self.survival_bins = (
                    survival_discretizer.compute_apply_discretizer(split_metadata)
                )
            elif self.survival_bins is not None:
                split_metadata = survival_discretizer.apply_discretizer(
                    split_metadata, self.survival_bins
                )
            self.label_cols = ["survival_group", self.label_cols[1], self.label_cols[0]]

        # Drop samples for which no data are available
        slides_ids = [
            os.path.splitext(feat_file)[0]
            for feat_dir in self.features_dirs
            for feat_file in os.listdir(feat_dir)
        ]
        split_metadata = split_metadata[
            split_metadata["slide_id"].isin(slides_ids)
        ].reset_index(drop=True)

        # [patch-label] Also drop slides whose annotation file is missing.
        if patch_metadata_dirs is not None and patch_label_method is not None:
            split_metadata = (
                SlideDataHandler.filter_slide_metadata_to_existing_patches(
                    split_metadata, patch_metadata_dirs, patch_label_method
                )
            )

        # Drop duplicates
        if drop_duplicates == "sample":
            split_metadata = split_metadata.drop_duplicates(
                "slide_id", keep="first"
            ).reset_index(drop=True)
        elif drop_duplicates == "case":
            split_metadata = split_metadata.drop_duplicates(
                "case_id", keep="first"
            ).reset_index(drop=True)
        else:
            raise ValueError(
                f"Unknown level for dropping duplicates: {drop_duplicates}"
            )
        self.split_metadata = split_metadata
        # Load patch metadata and data
        # [patch-label] ``load_patch_metadata`` now also returns patch_labels.
        (
            self.feature_indices,
            self.patch_ids,
            self.patch_labels,                       # [patch-label] added
            self.positions_x,
            self.positions_y,
        ) = SlideDataHandler.load_patch_metadata(
            self.split_metadata,
            patches_dirs,
            patch_filters,
            patch_metadata_dirs=patch_metadata_dirs,   # [patch-label]
            patch_label_method=patch_label_method,     # [patch-label]
        )
        # Drop bags with too many patches
        if max_bag_size is not None and max_bag_size > 0 or min_bag_size > 0:
            max_bag_size = (
                torch.inf
                if (max_bag_size is None or max_bag_size < 0)
                else max_bag_size
            )
            keep_slides = [
                idx
                for idx in self.split_metadata.index
                if min_bag_size <= len(self.patch_ids[idx]) <= max_bag_size
            ]
            print(
                f"Dropping {len(self.split_metadata) - len(keep_slides)} slides with more than {max_bag_size} "
                f"or fewer than {min_bag_size} patches."
            )
            self.split_metadata = self.split_metadata.iloc[keep_slides].reset_index(
                drop=True
            )
            self.feature_indices = [self.feature_indices[idx] for idx in keep_slides]
            self.patch_ids = [self.patch_ids[idx] for idx in keep_slides]
            self.patch_labels = [self.patch_labels[idx] for idx in keep_slides]  # [patch-label]
            self.positions_x = [self.positions_x[idx] for idx in keep_slides]
            self.positions_y = [self.positions_y[idx] for idx in keep_slides]
        if preload_features:
            print("Loading features into RAM")
            self.features = []
            for idx, row in tqdm(
                self.split_metadata.iterrows(), total=len(self.split_metadata)
            ):
                source_id, slide_id = row[["source_id", "slide_id"]]
                features_path = os.path.join(self.features_dirs[source_id], slide_id)
                self.features.append(
                    SlideDataHandler.load_features(
                        features_path, self.feature_indices[idx], self.patch_ids[idx]
                    )
                )
        else:
            self.features = None

        self.reference_target = 0 if reference_target is None else reference_target

    @staticmethod
    def bag_collate_fn(batch_list):
        """
        Custom collate function for this dataset.

        Note: ``patch_labels`` is concatenated alongside the upstream patch
        fields — added for Symbolic xMIL.
        """
        col_batch = {}
        concat_keys = [
            "features",
            "patch_ids",
            "positions_x",
            "positions_y",
            "patch_labels",   # [patch-label] added
        ]
        for key in batch_list[0].keys():
            if key in concat_keys:
                col_batch[key] = torch.concat([batch[key] for batch in batch_list])
            elif key == "targets":
                col_batch[key] = torch.stack([batch[key] for batch in batch_list])
            elif key == "sample_ids":
                col_batch[key] = {
                    col: [batch[key][col] for batch in batch_list]
                    for col in batch_list[0][key]
                }
            else:
                col_batch[key] = torch.tensor([batch[key] for batch in batch_list])
        return col_batch

    def get_metadata(self):
        return self.split_metadata

    def __len__(self):
        return len(self.split_metadata) * self.num_repetitions

    def __getitem__(self, idx):
        """
        :return: (dict)
            - 'features': (torch.Tensor) Filtered features of a slide. All features if bag_size is None, otherwise a
                fixed number of bag_size features, possibly zero-padded.
            - 'bag_size': (int) The number of features.
            - 'targets': (torch.Tensor) The prediction targets of the slide.
            - 'slide_id': (list) Identifier of the slide.
            - 'patch_ids': (torch.Tensor) Identifiers of the patches associated to the features; patch_ids[idx]
                corresponds to feature[idx].
            - 'patch_labels': (torch.Tensor) [patch-label] Integer class labels per patch,
                parallel to patch_ids.  Empty tensor when no patch-label method is set.
        """
        idx = idx // self.num_repetitions
        # Load relevant metadata
        source_id, slide_id = self.split_metadata.iloc[idx][["source_id", "slide_id"]]
        patch_ids = self.patch_ids[idx]
        positions_x = self.positions_x[idx]
        positions_y = self.positions_y[idx]
        patch_labels = self.patch_labels[idx]            # [patch-label] added
        targets = (
            torch.tensor(
                self.split_metadata.iloc[idx][self.label_cols].values.astype(float)
            )
            - self.reference_target
        )
        if not self.survival:
            targets = targets.long()

        # Load (filtered) features
        if self.features is None:
            features_path = os.path.join(self.features_dirs[source_id], slide_id)
            features = SlideDataHandler.load_features(
                features_path, self.feature_indices[idx], self.patch_ids[idx]
            )
        else:
            features = self.features[idx]
        if self.bag_size is not None and self.bag_size > 0:
            # [patch-label] sample_features now also samples patch_labels.
            features, patch_ids, positions_x, positions_y, patch_labels = (
                SlideDataHandler.sample_features(
                    features,
                    self.bag_size,
                    self.sort_sampled_patches,
                    patch_ids,
                    positions_x,
                    positions_y,
                    patch_labels=patch_labels,           # [patch-label]
                )
            )

        return {
            "features": features,
            "bag_size": len(features),
            "targets": targets,
            "patch_ids": patch_ids,
            "sample_ids": {"source_id": source_id, "slide_id": slide_id},
            "positions_x": positions_x,
            "positions_y": positions_y,
            "patch_labels": patch_labels,                # [patch-label] added
        }


class MILOmicsDataset(Dataset):
    """Compatibility placeholder for omics-only xMIL datasets."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MILOmicsDataset is not available in this Symbolic xMIL workspace."
        )
