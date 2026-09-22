"""Train a behavioral cloning (BC) policy on simulated expert trajectories.

This version:
- Logs per-epoch metrics to metrics_bc.json
- Saves train/test split sizes and class statistics
- Prints best test accuracy and where the model is saved
"""
from __future__ import annotations


import ast
import json
import random
from pathlib import Path


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Configuration
# -----------------------------
TRAJECTORIES_PATH: str = "data/trajectories_expert_5k.csv"

RANDOM_SEED = 42
TEST_RATIO = 0.2
BATCH_SIZE = 256
HIDDEN_DIMS = (256, 128)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
PATIENCE = 7  # early stopping
MODEL_SAVE_PATH = "bc_policy.pt"
METRICS_SAVE_PATH = "metrics_bc.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    df = pd.read_csv(path)
    states = np.vstack(df["state"].apply(ast.literal_eval).tolist())
    actions = df["action"].to_numpy(dtype=np.int64)
    query_ids = df["query_id"].to_numpy()
    return states, actions, query_ids


def split_by_query(query_ids: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(query_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_ratio))
    test_ids = set(unique[:n_test].tolist())
    train_mask = np.array([q not in test_ids for q in query_ids])
    return train_mask


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_actions: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main():
    set_seed(RANDOM_SEED)
    states, actions, query_ids = load_data(TRAJECTORIES_PATH)
    print(f"Loaded {len(states):,} transitions from {TRAJECTORIES_PATH}")
    print(
        f"State dim: {states.shape[1]}, "
        f"actions: {sorted(np.unique(actions).tolist())}, "
        f"device: {DEVICE}"
    )

    train_mask = split_by_query(query_ids, TEST_RATIO, RANDOM_SEED)
    X_train, y_train = states[train_mask], actions[train_mask]
    X_test, y_test = states[~train_mask], actions[~train_mask]
    print(f"Train transitions: {len(y_train):,}, test transitions: {len(y_test):,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_t = torch.from_numpy(y_test).long().to(DEVICE)

    num_actions = int(np.max(actions) + 1)
    model = BCPolicy(X_train.shape[1], HIDDEN_DIMS, num_actions).to(DEVICE)

    # Balanced class weights so the model does not collapse to the majority action.
    class_counts = np.bincount(y_train, minlength=num_actions)
    class_weights = torch.tensor(
        (class_counts.sum() / (num_actions * class_counts)).astype(np.float32),
        device=DEVICE,
    )
    print(f"Class counts (train): {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_logs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss = criterion(test_logits, y_test_t).item()
            test_acc = accuracy(test_logits, y_test_t)

        avg_train_loss = float(np.mean(train_losses))
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} — "
            f"train loss: {avg_train_loss:.4f}, "
            f"test loss: {test_loss:.4f}, "
            f"test acc: {test_acc:.4f}"
        )

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_accuracy": float(best_acc),
                },
                MODEL_SAVE_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best test acc: {best_acc:.4f} (epoch {best_epoch})"
                )
                break

    print(f"Best test accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Save metrics to JSON
    metrics = {
        "config": {
            "random_seed": RANDOM_SEED,
            "test_ratio": TEST_RATIO,
            "batch_size": BATCH_SIZE,
            "hidden_dims": list(HIDDEN_DIMS),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
        },
        "data": {
            "total_transitions": int(len(states)),
            "train_transitions": int(len(y_train)),
            "test_transitions": int(len(y_test)),
            "state_dim": int(states.shape[1]),
            "num_actions": int(num_actions),
            "class_counts_train": class_counts.tolist(),
            "class_weights": class_weights.cpu().numpy().tolist(),
        },
        "training": {
            "best_test_accuracy": float(best_acc),
            "best_epoch": int(best_epoch),
            "epochs_trained": int(epoch),
            "epoch_logs": epoch_logs,
        },
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()