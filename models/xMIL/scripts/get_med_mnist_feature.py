import argparse

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_PathMNIST_features(npz_path, device="cuda", batch_size=64):
    """Extract Virchow2 features from PathMNIST and return a class → tensor dict.
    PathMNIST has 9 classes (0-8). Concatenates train/val/test splits.
    """
    print(f"[1/5] Loading NPZ from {npz_path}")
    data = np.load(npz_path)
    print(f"      Keys: {list(data.keys())}")

    imgs = np.concatenate([data["train_images"], data["val_images"], data["test_images"]])
    labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]]).squeeze()
    print(f"      Total images: {imgs.shape}, labels: {labels.shape}, dtype: {imgs.dtype}")
    print(f"      Image value range: [{imgs.min()}, {imgs.max()}]")

    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.from_numpy(labels).long()
    print(f"      Tensor shape: {imgs.shape} (NCHW), labels unique: {labels.unique().tolist()}")

    dataset = TensorDataset(imgs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n[2/5] Loading Virchow2 model onto {device} ...")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).eval().to(device)
    print(f"      Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    print(f"      Transform config: input_size={config['input_size']}, mean={config['mean']}, std={config['std']}")

    print(f"\n[3/5] Extracting features ({len(dataset)} images, batch_size={batch_size}) ...")
    features, targets = [], []
    for batch_idx, (batch_imgs, batch_labels) in enumerate(tqdm(loader)):
        x = torch.stack([transform(img) for img in batch_imgs]).to(device)

        with torch.no_grad():
            out = model(x)  # (B, 261, 1280)

        if batch_idx == 0:
            print(f"\n      [debug] First batch — input: {x.shape}, model output: {out.shape}")

        cls_token = out[:, 0]               # (B, 1280)
        patch_tokens = out[:, 5:].mean(1)   # (B, 1280); skip 4 register tokens
        embedding = torch.cat([cls_token, patch_tokens], dim=-1)  # (B, 2560)

        if batch_idx == 0:
            print(f"      [debug] cls_token: {cls_token.shape}, patch_tokens: {patch_tokens.shape}, embedding: {embedding.shape}")

        features.append(embedding.cpu())
        targets.append(batch_labels)

    print(f"\n[4/5] Concatenating features ...")
    features = torch.cat(features)
    targets = torch.cat(targets)
    print(f"      features: {features.shape}, targets: {targets.shape}")

    print(f"\n[5/5] Building class → tensor dict ...")
    result = {t: features[targets == t] for t in range(9)}
    for cls_idx, tensor in result.items():
        print(f"      class {cls_idx}: {tensor.shape[0]} samples, feature dim {tensor.shape[1]}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Virchow2 features from PathMNIST.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to pathmnist_224.npz (MedMNIST).")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the class -> features dict (.pt).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    feat_dict = get_PathMNIST_features(
        npz_path=args.npz_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"\nDone. Saving ...")
    torch.save(feat_dict, args.output_path)
    print(f"Saved to {args.output_path}")