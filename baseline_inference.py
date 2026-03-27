"""Deterministic baseline rollout script for local or remote servers."""

from __future__ import annotations

import argparse
import json
from statistics import mean
from pathlib import Path
import sys

import requests

try:
    from .client import SupportOpsEnv
    from .evaluation import baseline_plan_for_task
    from .models import EpisodeExport
    from .task_catalog import ORDERED_TASK_IDS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from support_ops_env.client import SupportOpsEnv
    from support_ops_env.evaluation import baseline_plan_for_task
    from support_ops_env.models import EpisodeExport
    from support_ops_env.task_catalog import ORDERED_TASK_IDS


def run_task(base_url: str, task_id: str) -> dict[str, object]:
    client = SupportOpsEnv(base_url=base_url).sync()
    with client:
        result = client.reset(task_id=task_id, seed=7)
        for action in baseline_plan_for_task(task_id):
            result = client.step(action)
            if result.done:
                break
        state = client.state()

    episode = EpisodeExport(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    results = [run_task(args.base_url, task_id) for task_id in ORDERED_TASK_IDS]
    average_score = round(mean(item["score"] for item in results), 4)
    print(
        json.dumps(
            {"average_score": average_score, "tasks": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
