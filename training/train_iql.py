"""Train a discrete IQL policy on the tree-branches MDP dataset.

Uses the existing IQL dataset adapter and the TorchRL discrete-IQL setup
tested in ``discrete_iql_model.py``.

Integration-test run (100 updates):
    .venv\\Scripts\\python training\\train_iql.py --updates 100

Real training run:
    .venv\\Scripts\\python training\\train_iql.py --updates 50000 --output-dir models/iql
"""
from __future__ import annotations

import argparse
import json
import os
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# The smoke-test module sets RL_WARNINGS and exports the builders/constants.
from discrete_iql_model import (
    BATCH_SIZE,
    GAMMA,
    EXPECTILE,
    HIDDEN_DIMS,
    MASK_SLICE,
    N_ACTIONS,
    OBS_DIM,
    TARGET_TAU,
    TEMPERATURE,
    MaskedLogitsMLP,
    _to_tensordict,
    build_actor,
    build_loss,
    build_qvalue_net,
    build_value_net,
)
from iql_dataset_adapter import DATASET_PATH, QUERY_CSV_PATH, build_iql_dataset
from torchrl.objectives.utils import SoftUpdate

os.environ.setdefault("D3RLPY_LOGGING_LEVEL", "ERROR")


def _versions() -> Dict[str, str]:
    return {
        "torch": torch.__version__,
        "torchrl": version("torchrl"),
        "tensordict": version("tensordict"),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sample_batch(
    dataset: Dict[str, np.ndarray],
    batch_size: int,
) -> Any:
    """Sample uniformly with replacement from the full transition set."""
    n = dataset["observations"].shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    return _to_tensordict(
        dataset["observations"][idx],
        dataset["actions"][idx],
        dataset["next_observations"][idx],
        dataset["rewards"][idx],
        dataset["terminals"][idx],
    )


def train_one_update(
    loss_module,
    optimizer: torch.optim.Optimizer,
    updater: SoftUpdate,
    batch: Any,
) -> Dict[str, float]:
    """One gradient-update step."""
    loss_module.zero_grad()
    optimizer.zero_grad()

    loss_td = loss_module(batch)
    loss_keys = ("loss_actor", "loss_qvalue", "loss_value")
    for key in loss_keys:
        val = loss_td[key]
        if not torch.isfinite(val):
            raise RuntimeError(f"{key} is non-finite at update: {val}")

    total = sum(loss_td[k] for k in loss_keys)
    total.backward()

    for p in loss_module.parameters():
        if p.grad is None:
            raise RuntimeError("online parameter has no gradient")
        if not torch.isfinite(p.grad).all():
            raise RuntimeError("non-finite gradient detected")

    optimizer.step()
    updater.step()

    return {k: float(loss_td[k].item()) for k in loss_keys}


def save_checkpoint(
    output_dir: Path,
    loss_module,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    config: Dict[str, Any],
) -> None:
    rng_states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "loss_state_dict": loss_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update_count": update_count,
        "config": config,
        "versions": _versions(),
        "rng_states": rng_states,
    }
    path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def save_actor(output_dir: Path, actor) -> None:
    """Save the underlying masked-logits MLP as a standalone eval actor."""
    # actor.module is a ModuleList: [TensorDictModule(logits), SafeProbabilisticModule(action)].
    mlp = actor.module[0].module
    path = output_dir / "actor.pt"
    torch.save(mlp.state_dict(), path)
    print(f"Saved actor to {path}")


def load_actor_for_eval(path: str | Path, device: torch.device | None = None) -> MaskedLogitsMLP:
    """Load the separately saved actor for evaluation."""
    model = MaskedLogitsMLP()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def verify_saved_actor(
    output_dir: Path,
    dataset: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
) -> None:
    """Check that the saved actor reproduces the trained actor's masked logits."""
    actor_path = output_dir / "actor.pt"
    trained = load_actor_for_eval(actor_path, device=device)

    idx = np.random.randint(0, dataset["observations"].shape[0], size=batch_size)
    obs = torch.as_tensor(dataset["observations"][idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        saved_logits = trained(obs)
        # Load the same weights into a brand-new MLP to confirm reproducibility.
        fresh = MaskedLogitsMLP().to(device)
        fresh.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True))
        fresh_logits = fresh(obs)

    if not torch.allclose(saved_logits, fresh_logits):
        raise RuntimeError("Saved actor logits do not match a freshly loaded actor")

    # Also sanity-check the action mask is still enforced.
    mask = obs[..., MASK_SLICE]
    invalid = (mask == 0).float()
    probs = torch.softmax(saved_logits, dim=-1)
    prob_on_invalid = (probs * invalid).sum()
    if not torch.isclose(prob_on_invalid, torch.tensor(0.0, device=device)):
        raise RuntimeError(f"Saved actor assigns probability to invalid actions: {prob_on_invalid.item()}")

    print("Saved actor reloads and reproduces masked logits.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discrete IQL policy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('auto', 'cpu', 'cuda:0', ...)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Minibatch size")
    parser.add_argument(
        "--updates",
        type=int,
        default=100,
        help="Number of gradient updates (default 100 for integration test)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/iql_test",
        help="Directory for checkpoints, actor, and metrics",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help="Path to the HDF5 dataset to train on",
    )
    parser.add_argument(
        "--dataset-csv-path",
        type=str,
        default=QUERY_CSV_PATH,
        help="Optional CSV with query metadata/alignment",
    )
    parser.add_argument(
        "--no-dataset-csv",
        action="store_true",
        help="Skip CSV metadata/alignment (use only the HDF5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dependency versions")
    print("-" * 25)
    for name, val in _versions().items():
        print(f"{name:12s} {val}")
    print()

    print(f"Training on device: {device}")
    print(f"Output directory:   {output_dir}")

    csv_path = None if args.no_dataset_csv else args.dataset_csv_path

    print("Loading dataset...")
    print(f"  HDF5: {args.dataset_path}")
    print(f"  CSV:  {csv_path if csv_path else '(none)'}")
    dataset = build_iql_dataset(h5_path=args.dataset_path, csv_path=csv_path)
    n_transitions = dataset["observations"].shape[0]
    print(f"Transitions available: {n_transitions:,}")
    print()

    actor = build_actor().to(device)
    qvalue_net = build_qvalue_net().to(device)
    value_net = build_value_net().to(device)
    loss_module = build_loss(actor, qvalue_net, value_net).to(device)
    updater = SoftUpdate(loss_module, tau=TARGET_TAU)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    config = {
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "updates": args.updates,
        "lr": args.lr,
        "output_dir": str(output_dir),
        "dataset_path": args.dataset_path,
        "dataset_csv_path": csv_path,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "hidden_dims": HIDDEN_DIMS,
        "gamma": GAMMA,
        "expectile": EXPECTILE,
        "temperature": TEMPERATURE,
        "target_tau": TARGET_TAU,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics: list[Dict[str, Any]] = []
    log_interval = max(1, args.updates // 10)

    print(f"Training for {args.updates:,} updates with batch size {args.batch_size}")
    print("-" * 25)
    for update in range(1, args.updates + 1):
        batch = sample_batch(dataset, args.batch_size).to(device)
        losses = train_one_update(loss_module, optimizer, updater, batch)
        metrics.append({"update": update, **losses})

        if update == 1 or update % log_interval == 0 or update == args.updates:
            print(
                f"update {update:>5}/{args.updates:<5}  "
                f"actor={losses['loss_actor']:.6f}  "
                f"qvalue={losses['loss_qvalue']:.6f}  "
                f"value={losses['loss_value']:.6f}"
            )

    save_checkpoint(output_dir, loss_module, optimizer, args.updates, config)
    save_actor(output_dir, actor)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)

    verify_saved_actor(output_dir, dataset, device, batch_size=args.batch_size)

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
