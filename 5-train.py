"""Train and evaluate a CNN on PneumoniaMNIST chest X-rays.

Usage:
    python train.py --epochs 6 --outdir results

Downloads the dataset from Zenodo on first run (cached in ./data).
"""

import argparse
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from model import PneumoniaCNN

DATA_URLS = [
    "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
    "https://zenodo.org/record/4269852/files/pneumoniamnist.npz?download=1",
]
DATA_PATH = os.path.join("data", "pneumoniamnist.npz")


def download_data(path: str = DATA_PATH) -> str:
    """Download PneumoniaMNIST npz from Zenodo if not cached."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as r, open(
                path, "wb"
            ) as f:
                f.write(r.read())
            np.load(path)  # sanity check
            return path
        except Exception as exc:  # try mirror
            print(f"download failed from {url}: {exc}")
    raise RuntimeError("Could not download PneumoniaMNIST from any mirror.")


def load_split(path: str, split: str) -> TensorDataset:
    d = np.load(path)
    images = torch.from_numpy(d[f"{split}_images"]).float().unsqueeze(1) / 255.0
    labels = torch.from_numpy(d[f"{split}_labels"]).float().squeeze(1)
    return TensorDataset(images, labels)


def evaluate(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.sigmoid(logits).cpu())
            targets.append(y)
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(targets, preds),
        "roc_auc": roc_auc_score(targets, probs),
        "precision": precision_score(targets, preds),
        "recall": recall_score(targets, preds),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "probs": probs,
        "targets": targets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = download_data()
    train_ds = load_split(path, "train")
    val_ds = load_split(path, "val")
    test_ds = load_split(path, "test")

    # Class balance: PneumoniaMNIST is ~3:1 pneumonia:normal.
    train_labels = train_ds.tensors[1].numpy()
    pos_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    print(f"class balance pos_weight={pos_weight:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)

    model = PneumoniaCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        train_loss = total_loss / len(train_ds)

        # Validation loss and accuracy.
        model.eval()
        vloss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss += criterion(logits, y).item() * len(x)
                correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
                n += len(x)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss / n)
        history["val_acc"].append(correct / n)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={vloss / n:.4f} "
            f"val_acc={correct / n:.4f}"
        )

    metrics = evaluate(model, test_loader, device)
    print("test:", {k: v for k, v in metrics.items() if k not in ("probs", "targets")})

    os.makedirs(args.outdir, exist_ok=True)
    out_metrics = {k: v for k, v in metrics.items() if k not in ("probs", "targets")}
    out_metrics["config"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "parameters": n_params,
        "device": device,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(out_metrics, f, indent=2)

    # Training curves.
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, history["train_loss"], "o-", label="train loss")
    ax1.plot(epochs, history["val_loss"], "o-", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_acc"], "s--", color="green", label="val accuracy")
    ax2.set_ylabel("val accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=120)
    plt.close(fig)

    # Confusion matrix.
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["normal", "pneumonia"])
    ax.set_yticks([0, 1], ["normal", "pneumonia"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix (acc={metrics['accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # Sample predictions.
    test_images = test_ds.tensors[0].numpy()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test_images), size=16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    names = ["normal", "pneumonia"]
    for ax, i in zip(axes.flat, idx):
        ax.imshow(test_images[i, 0], cmap="gray")
        true = int(metrics["targets"][i])
        pred = int(metrics["probs"][i] >= 0.5)
        color = "green" if true == pred else "red"
        ax.set_title(
            f"true={names[true]} pred={names[pred]}", color=color, fontsize=9
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_predictions.png"), dpi=120)
    plt.close(fig)

    torch.save(model.state_dict(), os.path.join(args.outdir, "pneumonia_cnn.pt"))
    print(f"artifacts saved to {args.outdir}/")


if __name__ == "__main__":
    main()
