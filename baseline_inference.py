"""Baseline rollout script for local or remote servers."""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from pathlib import Path
import sys
from typing import Any

import requests

try:
    from .client import SupportOpsEnv
    from .evaluation import baseline_plan_for_task, run_in_process_baseline
    from .models import EpisodeExport, SupportOpsAction
    from .server.support_ops_env_environment import SupportOpsEnvironment
    from .task_catalog import ORDERED_TASK_IDS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from support_ops_env.client import SupportOpsEnv
    from support_ops_env.evaluation import baseline_plan_for_task, run_in_process_baseline
    from support_ops_env.models import EpisodeExport, SupportOpsAction
    from support_ops_env.server.support_ops_env_environment import SupportOpsEnvironment
    from support_ops_env.task_catalog import ORDERED_TASK_IDS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


def _episode_from_state(state: Any) -> EpisodeExport:
    return EpisodeExport(
        task_id=state.task_id,
        task_title=state.task_title,
        difficulty=state.difficulty,
        steps_taken=state.step_count,
        ticket_snapshot=state.ticket_snapshot,
        revealed_customer_facts=state.revealed_customer_facts,
        info_requests_made=state.info_requests_made,
        internal_notes=state.internal_notes,
        public_replies=state.public_replies,
        history=state.history,
        progress_score=state.progress_score,
    )


def _grade_episode(base_url: str, task_id: str, episode: EpisodeExport) -> dict[str, object]:
    grader_response = requests.post(
        f"{base_url}/grader",
        json={"task_id": task_id, "episode": episode.model_dump(mode="json")},
        timeout=30,
    )
    grader_response.raise_for_status()
    score_payload = grader_response.json()
    return {
        "task_id": task_id,
        "title": score_payload["title"],
        "score": score_payload["score"],
        "passed": score_payload["passed"],
        "steps_taken": episode.steps_taken,
    }


def run_task_deterministic(base_url: str, task_id: str) -> dict[str, object]:
    client = SupportOpsEnv(base_url=base_url).sync()
    with client:
        client.reset(task_id=task_id, seed=7)
        for action in baseline_plan_for_task(task_id):
            result = client.step(action)
            if result.done:
                break
        state = client.state()
    return _grade_episode(base_url, task_id, _episode_from_state(state))


def _build_openai_action(
    model_client: Any,
    model_name: str,
    observation: Any,
) -> SupportOpsAction:
    system_prompt = (
        "You are an automated support operations baseline agent. "
        "Return exactly one JSON object matching the action schema fields. "
        "Prioritize deterministic and safe behavior."
    )
    user_prompt = {
        "task_id": observation.task_id,
        "task_title": observation.task_title,
        "difficulty": observation.difficulty.value,
        "goal": observation.goal,
        "ticket_snapshot": observation.ticket_snapshot.model_dump(mode="json"),
        "outstanding_requirements": observation.outstanding_requirements,
        "revealed_customer_facts": observation.revealed_customer_facts,
        "available_operations": observation.metadata.get("available_operations", []),
    }
    completion = model_client.chat.completions.create(
        model=model_name,
        temperature=0,
        seed=7,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Choose the next action and output JSON only with keys from "
                    "SupportOpsAction.\n\n"
                    + json.dumps(user_prompt, indent=2)
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    return SupportOpsAction.model_validate(json.loads(raw))


def run_task_openai(base_url: str, task_id: str, model_name: str) -> dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai baseline mode.")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install dependencies first.")

    model_client = OpenAI(api_key=api_key)
    client = SupportOpsEnv(base_url=base_url).sync()
    with client:
        result = client.reset(task_id=task_id, seed=7)
        max_steps = 8
        for _ in range(max_steps):
            action = _build_openai_action(model_client, model_name, result.observation)
            result = client.step(action)
            if result.done:
                break
        state = client.state()
    return _grade_episode(base_url, task_id, _episode_from_state(state))


def run_baseline(base_url: str, mode: str, model_name: str) -> dict[str, object]:
    if mode == "in_process":
        baseline = run_in_process_baseline(SupportOpsEnvironment)
        return baseline.model_dump(mode="json")

    if mode == "openai":
        runner = lambda tid: run_task_openai(base_url, tid, model_name)
    else:
        runner = lambda tid: run_task_deterministic(base_url, tid)

    results = [runner(task_id) for task_id in ORDERED_TASK_IDS]
    average_score = round(mean(item["score"] for item in results), 4)
    return {"average_score": average_score, "tasks": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--mode",
        choices=["openai", "deterministic", "in_process"],
        default="openai",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BASELINE_OPENAI_MODEL", "gpt-4o-mini"),
    )
    args = parser.parse_args()

    result = run_baseline(args.base_url, args.mode, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
