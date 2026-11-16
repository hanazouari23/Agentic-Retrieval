# Discrete IQL model/loss smoke-test plan (revised)

## Goal
Add a TorchRL-based discrete IQL model and loss setup next to the existing `training/iql_dataset_adapter.py`, run only the smoke test, and report dependency versions and results. Do **not** implement a full training loop and do **not** modify the dataset or existing CQL code.

## 1. Dependency selection via Poetry

Current PyTorch: **2.10.0+cu126**.

- `torchrl==0.10.1` declares `torch>=2.1.0` and `tensordict>=0.10.0,<0.11.0`, so it is compatible with the installed PyTorch.
- Add it through Poetry so `pyproject.toml` and `poetry.lock` stay consistent:
  ```powershell
  poetry add "torchrl==0.10.1"
  ```
- After locking, **verify** that `torch`, `torchvision`, and `torchaudio` entries in `poetry.lock` still point to `2.10.0+cu126`. If Poetry tries to upgrade/downgrade PyTorch, abort and report the conflict. Do not silently bypass Poetry with `pip`.
- Run:
  ```powershell
  poetry install
  ```
- Confirm imports:
  ```python
  import torch, torchrl, tensordict
  print(torch.__version__, torchrl.__version__, tensordict.__version__)
  from torchrl.objectives import DiscreteIQLLoss
  from torchrl.objectives.utils import SoftUpdate
  ```

## 2. Verify the installed `DiscreteIQLLoss` API before wiring

Write a short throwaway check (e.g., `scripts/inspect_iql_api.py`) to confirm, for the **installed** version:

1. `DiscreteIQLLoss.__init__` accepts `qvalue_network` as a `TensorDictModule` **or** a `list[TensorDictModule]`, and accepts `gamma`, `num_qvalue_nets`, `temperature`, `expectile`, and `action_space`.
2. The loss creates `target_qvalue_network_params` (and ideally `target_value_network_params`) after construction.
3. With `action_space="categorical"`, actions of shape `(B,)` `int64` work.
4. `ProbabilisticActor` exposes logits, either via `actor.get_dist(batch)` or `actor.module(batch)["logits"]`.

Only proceed after these are confirmed. If a list of Q-networks is not supported, fall back to a single Q-network plus `num_qvalue_nets=2`.

## 3. Create `training/discrete_iql_model.py`

### 3.1 Hyperparameters and MLP builder

```python
OBS_DIM = 399
N_ACTIONS = 4
HIDDEN_DIMS = [256, 256]
GAMMA = 0.99
EXPECTILE = 0.7
TEMPERATURE = 3.0  # inverse temperature beta
BATCH_SIZE = 256
MASK_SLICE = slice(395, 399)
TARGET_TAU = 0.005


def _make_mlp(out_features: int) -> nn.Module:
    """Two 256-unit ReLU hidden layers, no output activation."""
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN_DIMS[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[1], out_features),
    )
```

### 3.2 Plain masked actor module

```python
class MaskedLogitsMLP(nn.Module):
    """Plain MLP that returns action logits masked by observation[:, 395:399]."""

    def __init__(self, obs_dim: int, n_actions: int, mask_slice: slice):
        super().__init__()
        self.net = _make_mlp(n_actions)
        self.mask_slice = mask_slice

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logits = self.net(observation)
        mask = observation[..., self.mask_slice]
        assert (mask.sum(dim=-1) > 0).all(), "encountered state with no valid action"
        return logits.masked_fill(mask == 0, float("-inf"))


actor_module = TensorDictModule(
    MaskedLogitsMLP(OBS_DIM, N_ACTIONS, MASK_SLICE),
    in_keys=["observation"],
    out_keys=["logits"],
)
actor = ProbabilisticActor(
    module=actor_module,
    in_keys=["logits"],
    out_keys=["action"],
    spec=CategoricalSpec(N_ACTIONS),
    distribution_class=torch.distributions.Categorical,
    return_log_prob=False,
)
```

### 3.3 Twin Q-networks and value network

```python
qvalue_nets = [
    SafeModule(
        _make_mlp(N_ACTIONS),
        in_keys=["observation"],
        out_keys=["state_action_value"],
    )
    for _ in range(2)
]

value_net = SafeModule(
    _make_mlp(1),
    in_keys=["observation"],
    out_keys=["state_value"],
)
```

### 3.4 Discrete IQL loss

```python
loss = DiscreteIQLLoss(
    actor_network=actor,
    qvalue_network=qvalue_nets,  # or single net if API check says list unsupported
    value_network=value_net,
    action_space="categorical",
    num_qvalue_nets=2,
    temperature=TEMPERATURE,
    expectile=EXPECTILE,
    gamma=GAMMA,
)
loss.make_value_estimator(gamma=GAMMA)
```

### 3.5 Explicit target Q-networks and `SoftUpdate`

The loss module automatically creates explicit target parameters (`target_qvalue_network_params`). Wire the TorchRL updater to them:

```python
from torchrl.objectives.utils import SoftUpdate

updater = SoftUpdate(loss, tau=TARGET_TAU)
```

Keep explicit references to the **online** trainable modules for gradient checks and future optimizers:

```python
online_params = (
    list(actor.parameters())
    + list(qvalue_nets[0].parameters())
    + list(qvalue_nets[1].parameters())
    + list(value_net.parameters())
)
```

### 3.6 TensorDict batch construction

```python
def _to_tensordict(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    reward: np.ndarray,
    terminal: np.ndarray,
) -> TensorDict:
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


def terminal_only_batch(dataset, batch_size=BATCH_SIZE):
    idx = np.where(dataset["terminals"] == 1)[0]
    idx = np.random.choice(idx, min(batch_size, len(idx)), replace=False)
    return _to_tensordict(
        dataset["observations"][idx],
        dataset["actions"][idx],
        dataset["next_observations"][idx],
        dataset["rewards"][idx],
        dataset["terminals"][idx],
    )


def nonterminal_only_batch(dataset, batch_size=BATCH_SIZE):
    idx = np.where(dataset["terminals"] == 0)[0]
    idx = np.random.choice(idx, min(batch_size, len(idx)), replace=False)
    return _to_tensordict(
        dataset["observations"][idx],
        dataset["actions"][idx],
        dataset["next_observations"][idx],
        dataset["rewards"][idx],
        dataset["terminals"][idx],
    )
```

### 3.7 Smoke-test function

```python
def _check_finite_losses(loss_td: TensorDict) -> dict[str, float]:
    out = {}
    for key in ("loss_actor", "loss_qvalue", "loss_value"):
        val = loss_td[key]
        assert torch.isfinite(val), f"{key} is non-finite: {val}"
        out[key] = val.item()
    return out


def _check_online_gradients(online_modules: list[nn.Module]) -> None:
    for m in online_modules:
        for p in m.parameters():
            assert p.grad is not None, "online parameter has no gradient"
            assert torch.isfinite(p.grad).all(), "non-finite gradient"


def _check_action_mask(actor_module: TensorDictModule, batch: TensorDict) -> None:
    with torch.no_grad():
        td = actor_module(batch)
        logits = td["logits"]
        probs = torch.softmax(logits, dim=-1)
        mask = batch["observation"][..., MASK_SLICE]
        invalid = (mask == 0).float()
        assert torch.allclose(probs * invalid, torch.zeros_like(probs)), \
            "invalid actions have non-zero probability"


def smoke_test() -> None:
    _report_versions()
    dataset = build_iql_dataset()
    print(f"Transitions available: {dataset['observations'].shape[0]:,}")

    actor = build_actor()
    qvalue_nets = build_q_nets()
    value_net = build_value_net()
    loss = build_loss(actor, qvalue_nets, value_net)
    updater = SoftUpdate(loss, tau=TARGET_TAU)

    online_modules = [actor, qvalue_nets[0], qvalue_nets[1], value_net]

    for name, batch_fn in [
        ("terminal-only", terminal_only_batch),
        ("nonterminal-only", nonterminal_only_batch),
    ]:
        print(f"\n{name} batch")
        batch = batch_fn(dataset)
        loss_td = loss(batch)
        losses = _check_finite_losses(loss_td)
        for k, v in losses.items():
            print(f"  {k}: {v:.6f}")

        total = sum(losses.values())
        total.backward()
        _check_online_gradients(online_modules)
        print("  online gradients finite")

        updater.step()
        print("  SoftUpdate step succeeded")

    print("\nAction-mask check")
    mask_batch = nonterminal_only_batch(dataset, batch_size=64)
    _check_action_mask(actor.module, mask_batch)
    print("  invalid actions have zero probability")
```

## 4. Run and report

```powershell
.venv\Scripts\python training\discrete_iql_model.py
```

Expected output to report:
- `torch`, `torchrl`, and `tensordict` versions.
- `Transitions available: 37,184`.
- Finite `loss_actor`, `loss_qvalue`, `loss_value` for both terminal-only and nonterminal-only batches.
- Finite gradients for online actor/Q/value parameters.
- `SoftUpdate` step runs without error.
- Action-mask check passes: invalid actions receive zero probability.
- No NaNs produced by terminal zero next-observations.

## 5. Constraints / what not to touch

- Do not modify `training/iql_dataset_adapter.py`.
- Do not modify `training/train_cql.py` or any other existing training script.
- Do not modify the HDF5/CSV dataset files.
- Keep observations unchanged; the mask is read directly from indices `395:399`.
- Only add the new `training/discrete_iql_model.py` file and the Poetry lock changes.

## 6. Fallbacks

- If `DiscreteIQLLoss` in the installed version does **not** accept a list of Q-networks, pass a single `SafeModule` and set `num_qvalue_nets=2`.
- If `SoftUpdate` cannot find target parameters for `IQLLoss`, inspect `loss.target_qvalue_network_params` and create a manual `target_qvalue_nets = deepcopy(qvalue_nets); target_qvalue_nets.requires_grad_(False)` together with a manual `tau`-weighted copy. Report the API limitation.
- If `poetry add torchrl==0.10.1` conflicts with the pinned CUDA torch, report the resolver output instead of forcing an installation.
