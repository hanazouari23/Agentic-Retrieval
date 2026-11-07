"""Run an evaluation policy and produce MS MARCO run files and diagnostics."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running directly from agentsim_experiment/.
project_root = Path(__file__).resolve().parents[1]
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import build_agents
from env.action_registry import load_action_registry
from env.pipeline_env import MSMARCOOnlinePipelineEnv
from evaluation.io_utils import (
    load_eval_config,
    load_queries,
    load_qrels,
    load_behavior_actions,
    load_training_queries,
    select_held_out_queries,
    write_diagnostics,
    write_run_file,
)
from policy.load_cql import load_cql_policy, select_action
from policy.observation_builder import ObservationBuilder


def run_policy(
    queries: List[Dict[str, str]],
    agents: Dict[str, Any],
    observation_builder: ObservationBuilder,
    action_registry: Dict[int, List[str]],
    policy_name: str,
    policy_fn,
    max_rounds: int,
    run_path: str | Path,
    diagnostics_path: str | Path,
    max_rank: int = 1000,
) -> None:
    """Execute a policy over a query set and write run/diagnostics files."""
    logger = logging.getLogger(__name__)
    query_results: List[Tuple[str, List[Tuple[str, float]]]] = []
    diagnostics: List[dict] = []

    for query in queries:
        qid = query["qid"]
        env = MSMARCOOnlinePipelineEnv(
            query=query,
            agents=agents,
            observation_builder=observation_builder,
            action_registry=action_registry,
            max_rounds=max_rounds,
        )

        observation, info = env.reset(seed=42)
        try:
            action_id = policy_fn(qid, observation)
        except Exception as exc:
            logger.exception("Policy failed for query %s: %s", qid, exc)
            action_id = 4  # fallback full pipeline

        start_time = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(action_id)
        episode_latency = time.perf_counter() - start_time

        ranked = info.get("ranked_passages", [])
        query_results.append((qid, ranked))

        # Per-tool latencies stored on the state.
        latencies = {
            tool: sum(values) if isinstance(values, list) else float(values)
            for tool, values in env.state.items()
            if tool.startswith("latency_")
        }

        diagnostics.append({
            "qid": qid,
            "action_id": action_id,
            "pipeline_steps": info.get("pipeline_steps", []),
            "agent_calls": info.get("agent_calls", {}),
            "num_ranked": len(ranked),
            "latencies": latencies,
            "episode_latency_seconds": episode_latency,
            "failed": info.get("failed", False),
            "failure_reason": info.get("failure_reason"),
        })

    write_run_file(run_path, query_results, policy_name, max_rank=max_rank)
    write_diagnostics(diagnostics_path, diagnostics)
    logger.info("Wrote %s run to %s and diagnostics to %s", policy_name, run_path, diagnostics_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavior or CQL policy on MS MARCO held-out queries."
    )
    parser.add_argument(
        "--config",
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml",
    )
    parser.add_argument(
        "--policy",
        choices=["cql", "behavior", "both"],
        default="both",
        help="Which policy to run",
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)
    paths = config["paths"]
    outputs = config["outputs"]
    eval_cfg = config.get("evaluation", {})
    agent_cfg = config.get("agents", {})
    behavior_cfg = config.get("behavior", {})

    logging.basicConfig(
        level=logging.getLevelName(config.get("logging", {}).get("level", "INFO")),
        format=config.get("logging", {}).get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading action registry from %s", paths["task_config"])
    action_registry = load_action_registry(paths["task_config"])

    logger.info("Loading observation builder")
    observation_builder = ObservationBuilder.from_task_config(paths["task_config"])
    max_rounds = observation_builder.max_rounds

    logger.info("Loading agents")
    agents = build_agents(config)

    logger.info("Loading queries and qrels")
    queries = load_queries(paths["queries"])
    qrels = load_qrels(paths["qrels"])
    behavior_actions = load_behavior_actions(
        behavior_cfg.get("episode_manifest", "episode_manifest.jsonl"),
        fallback_action=behavior_cfg.get("fallback_action", 4),
    )

    training_queries = load_training_queries(
        behavior_cfg.get("episode_manifest", "episode_manifest.jsonl"),
        split="train",
    )
    selected_queries = select_held_out_queries(
        queries,
        qrels,
        behavior_actions,
        num_queries=eval_cfg.get("num_queries"),
        seed=eval_cfg.get("seed", 42),
        exclude_training=training_queries,
    )
    logger.info("Selected %d held-out queries", len(selected_queries))

    if args.policy in ("cql", "both"):
        model_path = Path(paths["cql_model"])
        if not model_path.exists():
            logger.error(
                "CQL model not found at %s. Place a trained .d3 model there or "
                "change paths.cql_model in %s.",
                model_path,
                args.config,
            )
            if args.policy == "cql":
                raise FileNotFoundError(f"CQL model not found: {model_path}")
        else:
            logger.info("Loading CQL model from %s", model_path)
            cql = load_cql_policy(model_path)

            def cql_policy_fn(qid: str, observation: np.ndarray) -> int:
                return select_action(cql, observation)

            run_policy(
                queries=selected_queries,
                agents=agents,
                observation_builder=observation_builder,
                action_registry=action_registry,
                policy_name="cql_policy",
                policy_fn=cql_policy_fn,
                max_rounds=max_rounds,
                run_path=outputs["cql_run"],
                diagnostics_path=outputs["cql_diagnostics"],
                max_rank=eval_cfg.get("max_rank", 1000),
            )

    if args.policy in ("behavior", "both"):
        fallback = behavior_cfg.get("fallback_action", 4)

        def behavior_policy_fn(qid: str, observation: np.ndarray) -> int:
            # The episode manifest keys behavior actions by raw query text.
            query_text = next(
                (q["text"] for q in selected_queries if q["qid"] == qid), ""
            )
            return behavior_actions.get(query_text, fallback)

        run_policy(
            queries=selected_queries,
            agents=agents,
            observation_builder=observation_builder,
            action_registry=action_registry,
            policy_name="behavior_policy",
            policy_fn=behavior_policy_fn,
            max_rounds=max_rounds,
            run_path=outputs["behavior_run"],
            diagnostics_path=outputs["behavior_diagnostics"],
            max_rank=eval_cfg.get("max_rank", 1000),
        )


if __name__ == "__main__":
    main()
