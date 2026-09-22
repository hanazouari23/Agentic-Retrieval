r"""Evaluate a trained BC policy.

Usage:
    .venv\Scripts\python evaluate_bc_policy.py
    # or
    poetry run python evaluate_bc_policy.py

This script:
- Loads bc_policy.pt and data/trajectories_expert_5k.csv
- Reconstructs the same train/test split used in training
- Reports classification diagnostics (accuracy, per-step accuracy,
  per-action precision/recall/F1, confusion matrix)
- Checks whether predicted actions respect the valid-action mask
- Compares the policy against simple baselines (always STOP, majority class)
- Saves everything to eval_bc_policy.json
"""
from __future__ import annotations

import ast
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Configuration (must match training)
# -----------------------------
TRAJECTORIES_PATH: str = "data/trajectories_expert_5k.csv"
MODEL_PATH: str = "bc_policy.pt"
EVAL_SAVE_PATH: str = "eval_bc_policy.json"

RANDOM_SEED = 42
TEST_RATIO = 0.2
BATCH_SIZE = 256
HIDDEN_DIMS = (256, 128)

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
    steps = df["step"].to_numpy(dtype=np.int64) if "step" in df.columns else None
    return states, actions, query_ids, steps


def split_by_query(query_ids: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(query_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_ratio))
    test_ids = set(unique[:n_test].tolist())
    train_mask = np.array([q not in test_ids for q in query_ids])
    return train_mask, test_ids


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


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    metrics = {}
    for c in range(num_classes):
        tp = int(np.logical_and(y_true == c, y_pred == c).sum())
        fp = int(np.logical_and(y_true != c, y_pred == c).sum())
        fn = int(np.logical_and(y_true == c, y_pred != c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def valid_action_violations(states: np.ndarray, preds: np.ndarray):
    """The last 4 state dims are the valid-action mask [QR, RR, PRF, STOP]."""
    masks = states[:, 395:399]
    valid_for_pred = masks[np.arange(len(preds)), preds]
    n_violations = int((valid_for_pred < 0.5).sum())
    return n_violations, n_violations / len(preds)


def trajectory_level_exact_match(query_ids, steps, y_true, y_pred):
    """Fraction of queries where every predicted action matches the expert."""
    qids = np.unique(query_ids)
    matches = 0
    for qid in qids:
        mask = query_ids == qid
        if np.array_equal(y_true[mask], y_pred[mask]):
            matches += 1
    return matches / len(qids) if qids.size else 0.0


def main():
    set_seed(RANDOM_SEED)
    states, actions, query_ids, steps = load_data(TRAJECTORIES_PATH)
    print(f"Loaded {len(states):,} transitions from {TRAJECTORIES_PATH}")

    train_mask, test_ids = split_by_query(query_ids, TEST_RATIO, RANDOM_SEED)
    X_train, y_train = states[train_mask], actions[train_mask]
    X_test, y_test = states[~train_mask], actions[~train_mask]
    qids_test = query_ids[~train_mask]
    steps_test = steps[~train_mask] if steps is not None else None

    print(f"Train transitions: {len(y_train):,}, test transitions: {len(y_test):,}")
    print(f"Held-out test queries: {len(test_ids)}")

    num_actions = int(np.max(actions) + 1)
    model = BCPolicy(X_train.shape[1], HIDDEN_DIMS, num_actions).to(DEVICE)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_bc.py first.")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "unknown")
    else:
        model.load_state_dict(ckpt)
        epoch = "unknown"
    print(f"Loaded model from {MODEL_PATH} (trained epoch {epoch})")

    model.eval()

    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_t = torch.from_numpy(y_test).long().to(DEVICE)

    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        overall_acc = accuracy(logits, y_test_t)

    print(f"\nOverall test accuracy: {overall_acc:.4f}")

    # Baselines
    majority_class = int(Counter(y_train).most_common(1)[0][0])
    majority_acc = float(np.mean(y_test == majority_class))
    stop_acc = float(np.mean(y_test == 3))
    print(f"Majority-class baseline accuracy: {majority_acc:.4f} (class {majority_class})")
    print(f"Always-STOP baseline accuracy:    {stop_acc:.4f}")

    # Per-step accuracy
    if steps_test is not None:
        print("\nPer-step test accuracy:")
        for step in sorted(np.unique(steps_test)):
            mask = steps_test == step
            step_acc = float(np.mean(preds[mask] == y_test[mask]))
            print(f"  step {step}: {step_acc:.4f} ({int(mask.sum())} transitions)")

    # Per-action metrics
    print("\nPer-action metrics on test set:")
    pcm = per_class_metrics(y_test, preds, num_actions)
    for a, m in pcm.items():
        print(
            f"  action {a}: precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, f1={m['f1']:.4f} "
            f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})"
        )

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, num_actions)
    print("\nConfusion matrix (rows=expert, cols=policy):")
    header = "       " + " ".join(f"{a:>5}" for a in range(num_actions))
    print(header)
    for a in range(num_actions):
        row = " ".join(f"{cm[a, b]:>5}" for b in range(num_actions))
        print(f"  {a:>3} | {row}")

    # Valid-action mask violations
    n_viol, viol_rate = valid_action_violations(X_test, preds)
    print(f"\nValid-action mask violations: {n_viol}/{len(y_test)} ({viol_rate:.4f})")

    # Trajectory-level exact match
    traj_em = trajectory_level_exact_match(qids_test, steps_test, y_test, preds)
    print(f"Trajectory-level exact-match rate: {traj_em:.4f}")

    # Action distribution comparison
    expert_counts = Counter(y_test.tolist())
    policy_counts = Counter(preds.tolist())
    total = len(y_test)
    expert_dist = {int(a): expert_counts.get(a, 0) / total for a in range(num_actions)}
    policy_dist = {int(a): policy_counts.get(a, 0) / total for a in range(num_actions)}
    l1_diff = sum(abs(expert_dist[a] - policy_dist[a]) for a in range(num_actions)) / 2
    print("\nAction distribution on test set:")
    print("Action | Expert | Policy")
    for a in range(num_actions):
        print(f"  {a:6d} | {expert_dist[a]:.4f} | {policy_dist[a]:.4f}")
    print(f"Distribution L1 distance: {l1_diff:.4f}")

    # Save evaluation results
    eval_results = {
        "model_path": MODEL_PATH,
        "data_path": TRAJECTORIES_PATH,
        "config": {
            "random_seed": RANDOM_SEED,
            "test_ratio": TEST_RATIO,
            "hidden_dims": list(HIDDEN_DIMS),
        },
        "data_stats": {
            "total_transitions": int(len(states)),
            "train_transitions": int(len(y_train)),
            "test_transitions": int(len(y_test)),
            "num_actions": int(num_actions),
            "num_test_queries": len(test_ids),
        },
        "overall_test_accuracy": float(overall_acc),
        "baseline_accuracies": {
            "majority_class": majority_acc,
            "always_stop": stop_acc,
        },
        "per_step_accuracy": (
            {
                int(step): float(np.mean(preds[steps_test == step] == y_test[steps_test == step]))
                for step in sorted(np.unique(steps_test))
            }
            if steps_test is not None
            else {}
        ),
        "per_action_metrics": {str(k): v for k, v in pcm.items()},
        "confusion_matrix": cm.tolist(),
        "valid_action_violations": n_viol,
        "valid_action_violation_rate": viol_rate,
        "trajectory_exact_match_rate": traj_em,
        "action_distribution_test": {
            "expert": expert_dist,
            "policy": policy_dist,
        },
        "action_distribution_l1_distance": l1_diff,
    }

    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {EVAL_SAVE_PATH}")


if __name__ == "__main__":
    main()
