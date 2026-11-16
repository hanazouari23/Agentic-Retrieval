"""Discrete IQL model/loss smoke test using TorchRL.

Builds a masked categorical actor, twin Q-networks, and a value network,
wires them into ``torchrl.objectives.DiscreteIQLLoss``, and runs a
forward/backward smoke test on terminal-only and nonterminal-only minibatches
from the existing IQL dataset adapter.
"""
from __future__ import annotations

import os
from importlib.metadata import version
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import Categorical as CategoricalSpec
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives import DiscreteIQLLoss
from torchrl.objectives.utils import SoftUpdate

from iql_dataset_adapter import build_iql_dataset

# Keep d3rlpy quiet when the adapter imports it.
os.environ.setdefault("D3RLPY_LOGGING_LEVEL", "ERROR")
# Suppress TorchRL's "no target updater associated" warning once we create one.
os.environ["RL_WARNINGS"] = "False"

OBS_DIM: int = 399
N_ACTIONS: int = 4
HIDDEN_DIMS: list[int] = [256, 256]
MASK_SLICE = slice(395, 399)

GAMMA: float = 0.99
EXPECTILE: float = 0.7
TEMPERATURE: float = 3.0  # inverse temperature beta
TARGET_TAU: float = 0.005

BATCH_SIZE: int = 256


def _report_versions() -> Dict[str, str]:
    """Return dependency versions for reporting."""
    versions = {
        "torch": torch.__version__,
        "torchrl": version("torchrl"),
        "tensordict": version("tensordict"),
    }
    for name, val in versions.items():
        print(f"{name:12s} {val}")
    return versions


def _make_mlp(out_features: int) -> nn.Module:
    """Two 256-unit ReLU hidden layers, no output activation."""
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN_DIMS[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[1], out_features),
    )


class MaskedLogitsMLP(nn.Module):
    """Plain MLP that returns action logits masked by obs[:, 395:399]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _make_mlp(N_ACTIONS)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logits = self.net(observation)
        mask = observation[..., MASK_SLICE]
        assert (mask.sum(dim=-1) > 0).all(), "state with no valid action"
        return logits.masked_fill(mask == 0, float("-inf"))


def build_actor() -> ProbabilisticActor:
    """Categorical actor whose logits are masked by the observation action mask."""
    actor_module = TensorDictModule(
        MaskedLogitsMLP(),
        in_keys=["observation"],
        out_keys=["logits"],
    )
    return ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=CategoricalSpec(N_ACTIONS),
        distribution_class=torch.distributions.Categorical,
        return_log_prob=False,
    )


def build_qvalue_net() -> SafeModule:
    """Single Q-network; ``DiscreteIQLLoss`` will create twin copies."""
    return SafeModule(
        _make_mlp(N_ACTIONS),
        in_keys=["observation"],
        out_keys=["state_action_value"],
    )


def build_value_net() -> SafeModule:
    """State-value network V(s)."""
    return SafeModule(
        _make_mlp(1),
        in_keys=["observation"],
        out_keys=["state_value"],
    )


def build_loss(
    actor: ProbabilisticActor,
    qvalue_net: SafeModule,
    value_net: SafeModule,
) -> DiscreteIQLLoss:
    """Build the discrete IQL loss.

    Note: in torchrl 0.10.1 ``gamma`` must be passed via
    ``make_value_estimator``, not the loss constructor.
    """
    loss = DiscreteIQLLoss(
        actor_network=actor,
        qvalue_network=qvalue_net,
        value_network=value_net,
        action_space="categorical",
        num_qvalue_nets=2,
        temperature=TEMPERATURE,
        expectile=EXPECTILE,
    )
    loss.make_value_estimator(gamma=GAMMA)
    return loss


def _to_tensordict(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    reward: np.ndarray,
    terminal: np.ndarray,
) -> TensorDict:
    """Convert numpy slices to the TensorDict format DiscreteIQLLoss expects."""
    terminal_t = torch.as_tensor(terminal, dtype=torch.float32).unsqueeze(-1)
    return TensorDict(
        {
            "observation": torch.as_tensor(obs, dtype=torch.float32),
            "action": torch.as_tensor(action, dtype=torch.long),
            ("next", "observation"): torch.as_tensor(next_obs, dtype=torch.float32),
            ("next", "reward"): torch.as_tensor(reward, dtype=torch.float32).unsqueeze(-1),
            ("next", "done"): terminal_t.bool(),
            ("next", "terminated"): terminal_t.bool(),
        },
        batch_size=[obs.shape[0]],
    )


def terminal_only_batch(
    dataset: Dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
) -> TensorDict:
    """Sample a minibatch of terminal (STOP) transitions."""
    idx = np.where(dataset["terminals"] == 1)[0]
    if len(idx) == 0:
        raise ValueError("No terminal transitions in dataset")
    idx = np.random.choice(idx, min(batch_size, len(idx)), replace=False)
    return _to_tensordict(
        dataset["observations"][idx],
        dataset["actions"][idx],
        dataset["next_observations"][idx],
        dataset["rewards"][idx],
        dataset["terminals"][idx],
    )


def nonterminal_only_batch(
    dataset: Dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
) -> TensorDict:
    """Sample a minibatch of nonterminal transitions."""
    idx = np.where(dataset["terminals"] == 0)[0]
    if len(idx) == 0:
        raise ValueError("No nonterminal transitions in dataset")
    idx = np.random.choice(idx, min(batch_size, len(idx)), replace=False)
    return _to_tensordict(
        dataset["observations"][idx],
        dataset["actions"][idx],
        dataset["next_observations"][idx],
        dataset["rewards"][idx],
        dataset["terminals"][idx],
    )


def _check_finite_losses(loss_td: TensorDict) -> Dict[str, torch.Tensor]:
    """Assert the three loss terms are finite and return them as tensors."""
    losses: Dict[str, torch.Tensor] = {}
    for key in ("loss_actor", "loss_qvalue", "loss_value"):
        val = loss_td[key]
        assert torch.isfinite(val), f"{key} is non-finite: {val}"
        losses[key] = val
    return losses


def _check_online_gradients(loss_module: DiscreteIQLLoss) -> None:
    """Check that every trainable online parameter has a finite gradient."""
    trainable = list(loss_module.parameters())
    assert len(trainable) > 0, "No trainable parameters found"
    for p in trainable:
        assert p.grad is not None, "online parameter has no gradient"
        assert torch.isfinite(p.grad).all(), "non-finite gradient detected"


def _check_target_params_frozen(loss_module: DiscreteIQLLoss) -> None:
    """Confirm target network parameters are not trainable."""
    target = loss_module.target_qvalue_network_params
    assert not target.requires_grad, "target Q parameters should be frozen"


def _clone_container(container) -> dict:
    """Snapshot a TensorDictParams container as a flat dict of cloned tensors."""
    return {
        k: v.clone()
        for k, v in container.items(True, True)
        if isinstance(v, torch.Tensor)
    }


def _container_changed(before: dict, after: dict) -> bool:
    """Return True if any leaf tensor changed between snapshots."""
    for key in before:
        if not torch.allclose(before[key], after[key]):
            return True
    return False


def _check_container_gradients(container, name: str) -> None:
    """Assert every trainable leaf in the container has a finite gradient."""
    for key, p in container.items(True, True):
        if not isinstance(p, torch.Tensor) or not p.is_leaf or not p.requires_grad:
            continue
        assert p.grad is not None, f"{name} parameter {key} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} parameter {key} has non-finite gradient"


def _optimizer_step_check(
    loss_module: DiscreteIQLLoss,
    updater: SoftUpdate,
    dataset: Dict[str, np.ndarray],
) -> None:
    """Perform one real optimizer step and verify parameter updates."""
    print("One optimizer step")
    print("-" * 25)

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)
    batch = nonterminal_only_batch(dataset)
    loss_td = loss_module(batch)
    losses = _check_finite_losses(loss_td)
    for key, val in losses.items():
        print(f"  {key}: {val.item():.6f}")

    loss_module.zero_grad()
    optimizer.zero_grad()
    total = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_value"]
    total.backward()

    _check_container_gradients(loss_module.actor_network_params, "actor")
    _check_container_gradients(loss_module.qvalue_network_params, "qvalue")
    _check_container_gradients(loss_module.value_network_params, "value")
    print("  actor/Q/value gradients finite")

    before_online = {
        "actor": _clone_container(loss_module.actor_network_params),
        "qvalue": _clone_container(loss_module.qvalue_network_params),
        "value": _clone_container(loss_module.value_network_params),
    }
    before_target = _clone_container(loss_module.target_qvalue_network_params)

    # Locate a representative weight key for each container.
    def _first_weight_key(container):
        for k in container.keys():
            if k[-1] == "weight":
                return k
        raise ValueError("No weight key found")

    q_weight_key = _first_weight_key(before_online["qvalue"])

    optimizer.step()

    after_online = {
        "actor": _clone_container(loss_module.actor_network_params),
        "qvalue": _clone_container(loss_module.qvalue_network_params),
        "value": _clone_container(loss_module.value_network_params),
    }
    after_target_during_step = _clone_container(loss_module.target_qvalue_network_params)

    assert _container_changed(before_online["actor"], after_online["actor"]), "actor parameters did not change"
    assert _container_changed(before_online["value"], after_online["value"]), "value parameters did not change"
    assert _container_changed(before_online["qvalue"], after_online["qvalue"]), "qvalue parameters did not change"

    # Twin Q-networks are stacked along the first dimension of the functional params.
    q_weight_before = before_online["qvalue"][q_weight_key]
    q_weight_after = after_online["qvalue"][q_weight_key]
    assert not torch.allclose(q_weight_after[0], q_weight_before[0]), "first Q-network did not change"
    assert not torch.allclose(q_weight_after[1], q_weight_before[1]), "second Q-network did not change"
    print("  actor and both online Q-networks changed after optimizer step")

    assert not _container_changed(before_target, after_target_during_step), (
        "target Q parameters changed during the optimizer step"
    )
    print("  target Q parameters unchanged during optimizer step")

    updater.step()

    after_target_after_update = _clone_container(loss_module.target_qvalue_network_params)
    after_online_after_update = {
        "actor": _clone_container(loss_module.actor_network_params),
        "qvalue": _clone_container(loss_module.qvalue_network_params),
        "value": _clone_container(loss_module.value_network_params),
    }

    assert _container_changed(before_target, after_target_after_update), (
        "target Q parameters did not change after SoftUpdate"
    )
    print("  target Q parameters changed after SoftUpdate")

    for module in after_online:
        assert not _container_changed(after_online[module], after_online_after_update[module]), (
            f"online {module} parameters changed during SoftUpdate"
        )
    print("  online parameters unchanged during SoftUpdate")
    print()


def _check_action_mask(actor: ProbabilisticActor, batch: TensorDict) -> None:
    """Verify that invalid actions receive zero probability mass."""
    with torch.no_grad():
        actor(batch)  # populates logits and sampled action
        logits = batch["logits"]
        probs = torch.softmax(logits, dim=-1)
        mask = batch["observation"][..., MASK_SLICE]
        invalid = (mask == 0).float()
        prob_on_invalid = (probs * invalid).sum()
        assert torch.isclose(prob_on_invalid, torch.tensor(0.0)), (
            f"invalid actions have total probability {prob_on_invalid.item()}"
        )


def smoke_test() -> None:
    """Run the smoke test and print results."""
    print("Dependency versions")
    print("-" * 25)
    _report_versions()
    print()

    print("Loading dataset...")
    dataset = build_iql_dataset()
    print(f"Transitions available: {dataset['observations'].shape[0]:,}")
    print(f"Terminal transitions:  {int(dataset['terminals'].sum()):,}")
    print(f"Nonterminal transitions: {int((dataset['terminals'] == 0).sum()):,}")
    print()

    actor = build_actor()
    qvalue_net = build_qvalue_net()
    value_net = build_value_net()
    loss_module = build_loss(actor, qvalue_net, value_net)
    updater = SoftUpdate(loss_module, tau=TARGET_TAU)

    _check_target_params_frozen(loss_module)

    for name, batch_fn in [
        ("terminal-only", terminal_only_batch),
        ("nonterminal-only", nonterminal_only_batch),
    ]:
        print(f"{name} batch")
        print("-" * 25)
        batch = batch_fn(dataset)
        loss_td = loss_module(batch)
        losses = _check_finite_losses(loss_td)
        for key, val in losses.items():
            print(f"  {key}: {val.item():.6f}")

        total = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_value"]
        loss_module.zero_grad()
        total.backward()
        _check_online_gradients(loss_module)
        print("  online gradients finite")

        updater.step()
        print("  SoftUpdate step succeeded")
        print()

    _optimizer_step_check(loss_module, updater, dataset)

    print("Action-mask check")
    print("-" * 25)
    mask_batch = nonterminal_only_batch(dataset, batch_size=128)
    _check_action_mask(actor, mask_batch)
    print("  invalid actions have zero probability")
    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    smoke_test()
