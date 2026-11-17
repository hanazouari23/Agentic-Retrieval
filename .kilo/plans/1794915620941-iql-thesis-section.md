# Plan: Fill IQL thesis subsection

## Goal
Replace the placeholder text in the thesis section
`\subsubsection{Implementation and Training Procedure}` (label `sec:iql-implementation`)
with a fact-filled version derived from the actual code.

## Context gathered

Inspected files:
- `training/train_iql.py`
- `training/discrete_iql_model.py`
- `training/iql_dataset_adapter.py`
- `generate_iql_traces.py`
- `src/simulation.py` (policy integration and validity-mask handling)
- `training/train_cql.py` (for comparison)
- `models/iql_base_5000_seed42/config.json`
- `pyproject.toml` / `poetry.lock` (dependency versions)
- TorchRL source for `DiscreteIQLLoss` (critic aggregation, advantage clipping)
- `.kilo/plans/1794753829695-discrete-iql-model-smoke-test.md` (transition count)
- `notebooks/training_dataset_statistics.ipynb` (episode count)

Key facts:
- Implementation: TorchRL 0.10.1 `DiscreteIQLLoss` + custom masked categorical actor.
- Dependency versions: torch 2.10.0+cu126, torchrl 0.10.1, tensordict 0.10.0; d3rlpy 2.8.1 used only to load the HDF5 replay buffer.
- Default dataset: `data/training/mdp_dataset_tree_branches_5000.h5`; optional CSV `data/training/trajectories_tree_branches_5000_terminal_reward.csv` for alignment/query IDs. HDF5 contains ~19,999 episodes / 37,184 transitions (same base file CQL can use, but CQL default in `train_cql.py` is a reward-variant file; reward/terminal fields themselves come from the HDF5 and match CQL when the same HDF5 is used).
- Networks: separate 2-layer 256-unit ReLU MLPs for actor (logits), Q (4 outputs), and value (1 output). Input dimension 399; action mask in observation indices 395:399.
- Hyperparameters: Adam lr 3e-4 (single optimizer over all parameters), batch size 256, gamma 0.99, expectile 0.7, temperature (β) 3.0, target soft-update τ 0.005, twin critics, min aggregation, advantage weights clamped to 100, 50,000 gradient updates, seed 42, no validation split.
- Deployment: saved `actor.pt` is loaded as `MaskedLogitsMLP`; evaluation is greedy argmax through `IQLPolicyModel.predict`; integrated into `Simulation` with `policy="cql"`; operational mask forces invalid logits to −∞, so argmax cannot pick them; if an invalid action is returned anyway the simulator logs a warning and executes it (no fallback/replacement).

## Action
Insert the following completed LaTeX in place of the placeholder block.

```latex
\subsubsection{Implementation and Training Procedure}
\label{sec:iql-implementation}

\noindent
Training uses the same tree-branches HDF5 dataset as the CQL experiments:
`data/training/mdp_dataset_tree_branches_5000.h5`, loaded through
`d3rlpy`~2.8.1 as a `ReplayBuffer` and converted by
`training/iql_dataset_adapter.py` into flat transition arrays.
The adapter uses `d3rlpy`'s default `BasicTransitionPicker`, so the
observations, actions, rewards, next-observations, and terminal flags
are identical to those CQL sees when the same HDF5 is used.
The optional CSV
`data/training/trajectories_tree_branches_5000_terminal_reward.csv`
is used only for alignment verification and query-ID metadata; reward
values always come from the HDF5.
The base file contains approximately 19,999 episodes and 37,184
transitions (numbers taken from the existing smoke-test output and
dataset statistics notebook).
No observation normalization is applied; the 399-dimensional state is
used directly, with the last four dimensions (indices 395:399) encoding
the operational action mask.
For ablations, corrected reward variants derived from the same
trajectories are also used; whether the exact file reported in the CQL
subsection was used for every IQL run should be confirmed against the
experimental logs.

\begin{itemize}
    \item \textbf{Training input:}
The default training file is `data/training/mdp_dataset_tree_branches_5000.h5`
in `d3rlpy` ReplayBuffer format.
`training/iql_dataset_adapter.py` loads the buffer, optionally verifies
alignment against
`data/training/trajectories_tree_branches_5000_terminal_reward.csv`,
and emits dictionaries of NumPy arrays:
`observations`, `actions`, `rewards`, `next_observations`, `terminals`,
`action_masks`, and (when a CSV is supplied) `query_ids` and
`row_indices`.
Rewards and terminal indicators are taken from the HDF5, i.e. the same
source used by the CQL loader, so the labels match whenever the same
HDF5 file is used.
The base dataset is the same `mdp_dataset_tree_branches_5000.h5` file
that CQL can load; the specific reward-variant HDF5 used in the
reported CQL experiments should be cross-checked against the CQL
subsection because `train_cql.py` defaults to a stop-penalty variant.

\item \textbf{Model configuration and training loop:}
The IQL implementation is built on TorchRL~0.10.1
(`torchrl.objectives.DiscreteIQLLoss`), with PyTorch~2.10.0+cu126 and
TensorDict~0.10.0.
The four discrete actions (`QueryReform`, `Rerank`,
`PseudoRelevanceFeedback`, `STOP`) are handled as a categorical action
space (`action_space="categorical"`).
The actor is a `ProbabilisticActor` whose underlying `MaskedLogitsMLP`
produces four logits and forces invalid logits to $-\infty$ using the
observation mask slice `obs[..., 395:399]`.
The Q-network and value network are separate `SafeModule`s wrapping
plain MLPs; all three networks have two hidden layers of 256 units with
ReLU activation and no output activation.
Network inputs are the raw 399-dimensional observations.

A single Adam optimiser with learning rate $3\times10^{-4}$ optimises
all trainable parameters of the `DiscreteIQLLoss` module.
Mini-batches of size 256 are sampled uniformly with replacement from the
full transition set.
One gradient update is performed per iteration; the target Q-network is
updated after every update via TorchRL's `SoftUpdate` with Polyak
parameter $\tau=0.005$.
Training runs for a maximum of 50,000 gradient updates (100 in the
integration-test run), with loss values logged every
$\max(1, \text{updates} // 10)$ steps (i.e. every 5,000 steps for the
50,000-update run).
A single final checkpoint is saved; there is no validation-based
selection.
The random seed is 42.

\begin{table}[htbp]
    \centering
    \caption{Configuration of the IQL training procedure.}
    \label{tab:iql-training-configuration}
    \small
    \begin{tabular}{p{0.53\textwidth} p{0.37\textwidth}}
        \hline
        Parameter & Setting \\
        \hline
        Implementation and version & TorchRL `DiscreteIQLLoss` 0.10.1, PyTorch 2.10.0+cu126, TensorDict 0.10.0 \\
        Discrete-action implementation & Categorical actor with masked logits (`action_space="categorical"`) \\
        Q-network hidden-layer dimensions & [256, 256] \\
        Value-network hidden-layer dimensions & [256, 256] \\
        Policy-network hidden-layer dimensions & [256, 256] \\
        Hidden-layer activation & ReLU \\
        Observation preprocessing & None; raw 399-dim state, mask in indices 395:399 \\
        Optimiser(s) & Adam over all loss-module parameters \\
        Q-function learning rate & $3\times10^{-4}$ \\
        Value-function learning rate & $3\times10^{-4}$ \\
        Policy learning rate & $3\times10^{-4}$ \\
        Batch size & 256 \\
        Discount factor & 0.99 \\
        Number of critics & 2 \\
        Critic aggregation rule & Minimum over twin Q-networks \\
        Target-network update rule & SoftUpdate (Polyak) with $\tau=0.005$ after each update \\
        Expectile parameter $\tau$ & 0.7 \\
        Advantage inverse temperature $\beta$ & 3.0 \\
        Advantage-weight clipping & Clamped to maximum 100 inside `DiscreteIQLLoss` \\
        Maximum training budget & 50,000 gradient updates \\
        Updates per reporting epoch & 5,000 (one log line per 10\% of training) \\
        Random seed & 42 \\
        Validation partition & None (full dataset used) \\
        \hline
    \end{tabular}
\end{table}

\item \textbf{Saving the trained model:}
After training, `training/train_iql.py` writes the following to the
output directory:
`checkpoint.pt` (full training checkpoint containing
`loss_state_dict`, `optimizer_state_dict`, `update_count`, `config`,
dependency `versions`, and RNG states),
`actor.pt` (the standalone `MaskedLogitsMLP` state dict used for
evaluation),
`config.json` (the hyperparameter configuration), and
`metrics.json` (per-update loss history).
The policy, Q-function, value function, and training configuration are
therefore all recoverable from the checkpoint; the standalone
`actor.pt` is sufficient for rollout evaluation.

\item \textbf{Deploying the policy:}
For evaluation, `generate_iql_traces.py` loads `actor.pt` into a fresh
`MaskedLogitsMLP`, moves it to the target device, and wraps it in an
`IQLPolicyModel`.
The wrapper's `predict(obs)` method runs a forward pass and returns the
argmax action, so evaluation is deterministic and selects the most
probable action rather than sampling from the policy distribution.
The wrapper is passed to `Simulation` as `policy_model`, and rollouts
are generated with `policy="cql"`, which invokes
`Simulation._policy_cql`.
The operational validity mask is encoded in the last four dimensions of
the observation and enforced by the actor itself: invalid logits are
masked to $-\infty$ before the softmax/argmax, so the selected action
is always operationally valid under the learned model.
If an invalid action is ever returned (for example because the saved
weights differ from the trained actor), `Simulation._policy_cql` logs
a warning and returns the action unchanged; the simulator then executes
it without a fallback or replacement, which may lead to a repeated
agent call or a downstream error.
\end{itemize}
```

## Validation

1. Replace the placeholder block in the thesis source with the text above.
2. Search the section for any remaining `[To be ...]` or `[To be specified]` strings; there should be none.
3. Confirm the LaTeX compiles (no missing packages, `\textwidth` usage is fine inside table, math macros are correct).
4. Cross-check the final sentence about dataset identity with the CQL subsection; if the CQL experiments used a reward-variant file, clarify in the IQL text that the base HDF5 is shared but reward-variant ablations may differ.

## Open questions / out of scope

- Exact episode/transition counts for each reward-variant HDF5 were not
  recomputed here; the numbers above are for the base
  `mdp_dataset_tree_branches_5000.h5`.
- Whether the final reported IQL experiments used the optional CSV
  alignment (some saved configs have `"dataset_csv_path": null`) should
  be confirmed from the experimental logs.
