r"""Evaluate a trained Discrete CQL policy.

Usage:
    .venv\Scripts\python evaluate_cql_policy.py
    # or
    poetry run python evaluate_cql_policy.py

Mirrors the diagnostics reported for BC:
- overall test accuracy
- per-step accuracy
- per-action precision/recall/F1 and confusion matrix
- valid-action mask violations
- trajectory-level exact-match rate
- action-distribution distance vs. the expert
- baselines (majority class, always STOP)
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
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

# -----------------------------
# Configuration (must match training)
# -----------------------------
TRAJECTORIES_PATH: str = "data/trajectories_expert_5k.csv"
MODEL_PATH: str = "cql_policy.pt"
EVAL_SAVE_PATH: str = "eval_cql_policy.json"

RANDOM_SEED = 42
TEST_RATIO = 0.2
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0


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
    rewards = df["reward"].to_numpy(dtype=np.float32)
    terminals = df["terminal"].to_numpy(dtype=np.float32)
    timeouts = df["timeout"].to_numpy(dtype=np.float32)
    query_ids = df["query_id"].to_numpy()
    steps = df["step"].to_numpy(dtype=np.int64) if "step" in df.columns else None
    return states, actions, rewards, terminals, timeouts, query_ids, steps


def split_by_query(query_ids: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(query_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_ratio))
    test_ids = set(unique[:n_test].tolist())
    train_mask = np.array([q not in test_ids for q in query_ids])
    return train_mask, test_ids


def build_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    num_actions: int,
) -> MDPDataset:
    return MDPDataset(
        observations=states.astype(np.float32),
        actions=actions.astype(np.int64),
        rewards=rewards.astype(np.float32),
        terminals=terminals.astype(np.float32),
        timeouts=timeouts.astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=num_actions,
    )


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
    """State indices 395:399 hold the valid-action mask [QR, RR, PRF, STOP]."""
    masks = states[:, 395:399]
    valid_for_pred = masks[np.arange(len(preds)), preds]
    n_violations = int((valid_for_pred < 0.5).sum())
    return n_violations, n_violations / len(preds)


def trajectory_level_exact_match(query_ids, y_true, y_pred):
    qids = np.unique(query_ids)
    matches = 0
    for qid in qids:
        mask = query_ids == qid
        if np.array_equal(y_true[mask], y_pred[mask]):
            matches += 1
    return matches / len(qids) if qids.size else 0.0


def main():
    set_seed(RANDOM_SEED)
    (
        states,
        actions,
        rewards,
        terminals,
        timeouts,
        query_ids,
        steps,
    ) = load_data(TRAJECTORIES_PATH)
    print(f"Loaded {len(states):,} transitions from {TRAJECTORIES_PATH}")

    train_mask, test_ids = split_by_query(query_ids, TEST_RATIO, RANDOM_SEED)
    X_train, y_train = states[train_mask], actions[train_mask]
    X_test, y_test = states[~train_mask], actions[~train_mask]
    rewards_test = rewards[~train_mask]
    qids_test = query_ids[~train_mask]
    steps_test = steps[~train_mask] if steps is not None else None

    print(f"Train transitions: {len(y_train):,}, test transitions: {len(y_test):,}")
    print(f"Held-out test queries: {len(test_ids)}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_cql.py first.")

    num_actions = int(np.max(actions) + 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading CQL model on device: {device}")

    config = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        encoder_factory=VectorEncoderFactory(hidden_units=list(HIDDEN_DIMS)),
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        alpha=CQL_ALPHA,
    )
    algo = config.create(device=device)

    # Build the network architecture before loading weights.
    test_dataset = build_dataset(
        X_test, y_test, rewards_test, terminals[~train_mask], timeouts[~train_mask], num_actions
    )
    algo.build_with_dataset(test_dataset)
    algo.load_model(MODEL_PATH)

    preds = algo.predict(X_test.astype(np.float32))
    preds = np.asarray(preds).reshape(-1)

    overall_acc = float(np.mean(preds == y_test))
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
    for a in range(num_classes := num_actions):
        row = " ".join(f"{cm[a, b]:>5}" for b in range(num_classes))
        print(f"  {a:>3} | {row}")

    # Valid-action mask violations
    n_viol, viol_rate = valid_action_violations(X_test, preds)
    print(f"\nValid-action mask violations: {n_viol}/{len(y_test)} ({viol_rate:.4f})")

    # Trajectory-level exact match
    traj_em = trajectory_level_exact_match(qids_test, y_test, preds)
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

    eval_results = {
        "model_path": MODEL_PATH,
        "data_path": TRAJECTORIES_PATH,
        "config": {
            "random_seed": RANDOM_SEED,
            "test_ratio": TEST_RATIO,
            "hidden_dims": list(HIDDEN_DIMS),
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "cql_alpha": CQL_ALPHA,
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
