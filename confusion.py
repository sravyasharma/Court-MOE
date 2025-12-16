import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# =========================================================
# CONFIG (MATCHES TRAINING)
# =========================================================
EXPERT_DIR = "Experts/experts_kfold_final"
OUTPUT_DIR = "confusion_matrices"

EMBEDDINGS = {
    "supreme":  "encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
    "high":     "encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
    "tribunal": "encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
    "district": "encoding/metadata_augmented_v3_district_daily.pth",
    "daily":    "encoding/metadata_augmented_v3_district_daily.pth",
}

COURT_INDEX = {
    "supreme": 0,
    "high": 1,
    "district": 2,
    "tribunal": 3,
    "daily": 4,
}

BATCH_SIZE = 1024
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Accepted", "Rejected"]


# =========================================================
# LABEL HANDLING (MATCHES TRAINING)
# =========================================================
def label_to_binary(label):
    if isinstance(label, str):
        label = label.lower().strip()
        if label in ["accepted", "accept", "1", "true", "yes"]:
            return 1
        elif label in ["rejected", "reject", "0", "false", "no"]:
            return 0
        else:
            raise ValueError(f"Unknown label string: {label}")
    return int(label)


# =========================================================
# DATASET
# =========================================================
class EmbeddingDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# MODEL (EXACT MATCH TO ExpertNet)
# =========================================================
class ExpertNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.ln = torch.nn.LayerNorm(in_dim)
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.ln(x)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# =========================================================
# SAFE CHECKPOINT LOADER (DataParallel / EMA / SWA SAFE)
# =========================================================
def load_expert_weights_safely(model, state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)


# =========================================================
# LOAD EMBEDDINGS
# =========================================================
def load_embeddings(path: str, court: str) -> Tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    court_idx = COURT_INDEX[court]

    X, y = [], []
    for item in obj:
        if item["court_type_idx"] != court_idx:
            continue
        X.append(torch.tensor(item["embeddings"], dtype=torch.float32))
        y.append(label_to_binary(item["label"]))

    print(f"✅ {court.upper()} loaded: {len(y)} samples")
    return torch.stack(X), torch.tensor(y, dtype=torch.long)


# =========================================================
# CONFUSION MATRIX
# =========================================================
@torch.no_grad()
def compute_confusion_matrix(model, loader, threshold):
    model.eval()
    probs, targets = [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        targets.append(yb.numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


# =========================================================
# PNG HELPERS
# =========================================================
def normalize_confusion(tp, tn, fp, fn):
    cm = np.array([[tp, fn], [fp, tn]], dtype=float)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


def plot_confusion_png(cm_norm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm)

    ax.set_title(title, fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for court in ["supreme", "high", "tribunal", "district", "daily"]:
        print(f"\n{'='*70}")
        print(f"⚖️  {court.upper()} — CONFUSION MATRICES")
        print(f"{'='*70}")

        X, y = load_embeddings(EMBEDDINGS[court], court)
        loader = DataLoader(
            EmbeddingDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        for fold in range(1, 4):
            ckpt = torch.load(
                os.path.join(EXPERT_DIR, court, f"{court}_fold{fold}.pt"),
                map_location=DEVICE
            )

            model = ExpertNet(ckpt["config"]["in_dim"]).to(DEVICE)
            load_expert_weights_safely(model, ckpt["model_state_dict"])

            cm = compute_confusion_matrix(model, loader, ckpt["metrics"]["threshold"])
            print(f"Fold {fold} → {cm}")

            # Save JSON
            json_path = os.path.join(OUTPUT_DIR, f"{court}_fold{fold}.json")
            with open(json_path, "w") as f:
                json.dump(cm, f, indent=2)

            # Save PNG
            cm_norm = normalize_confusion(**cm)
            png_path = os.path.join(OUTPUT_DIR, f"{court}_fold{fold}_confusion.png")
            title = f"{court.capitalize()} Expert — Fold {fold}\nNormalized Confusion Matrix"
            plot_confusion_png(cm_norm, title, png_path)

            print(f"🖼️  Saved PNG → {png_path}")

        print("✔ Done")


if __name__ == "__main__":
    main()
