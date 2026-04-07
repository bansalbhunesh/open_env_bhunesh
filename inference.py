"""
Inference script for the Support Ops OpenEnv environment.

Uses the OpenAI Client with environment variables:
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - model identifier
    HF_TOKEN      - Hugging Face / API key

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (mandatory per hackathon rules)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# The environment server URL (local or HF Space)
ENV_SERVER_URL: str = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

BENCHMARK: str = "support_ops_env"

# ---------------------------------------------------------------------------
# OpenAI client (all LLM calls go through this)
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Task list (matches the environment's task catalog)
# ---------------------------------------------------------------------------
TASK_IDS: List[str] = [
    "password_reset_lockout",
    "duplicate_invoice_refund",
    "gdpr_export_incident",
]

MAX_STEPS_PER_TASK = 8


# ── structured logging helpers ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── helpers ────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict | None = None) -> dict:
    url = f"{ENV_SERVER_URL}{endpoint}"
    resp = requests.post(url, json=payload or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _build_system_prompt() -> str:
    return (
        "You are an expert support operations agent. You handle live customer "
        "tickets by triaging them, gathering evidence, writing internal notes, "
        "sending customer replies, and resolving or escalating cases.\n\n"
        "You MUST return exactly one JSON object with these fields:\n"
        "- operation: one of triage, request_customer_info, add_internal_note, send_reply, resolve\n"
        "- priority: (for triage) one of low, medium, high, urgent\n"
        "- queue: (for triage) one of customer_support, finance_ops, privacy_response\n"
        "- category: (for triage) one of access_management, billing_and_refunds, security_and_privacy\n"
        "- tags: list of short lowercase kebab-case tags\n"
        "- requested_fields: list of field names to request from the customer\n"
        "- internal_note: text for internal analyst notes\n"
        "- response_text: customer-facing reply text\n"
        "- resolution_code: one of guidance_provided, refund_review_opened, escalated_to_privacy_response\n"
        "- resolution_summary: short summary of the resolution\n\n"
        "Include only the fields relevant to the chosen operation. "
        "Return valid JSON only, no markdown, no explanation."
    )


def _build_user_prompt(observation: dict) -> str:
    context = {
        "task_id": observation.get("task_id", ""),
        "task_title": observation.get("task_title", ""),
        "difficulty": observation.get("difficulty", ""),
        "goal": observation.get("goal", ""),
        "customer_message": observation.get("customer_message", ""),
        "customer_profile": observation.get("customer_profile", {}),
        "knowledge_base": observation.get("knowledge_base", []),
        "ticket_snapshot": observation.get("ticket_snapshot", {}),
        "revealed_customer_facts": observation.get("revealed_customer_facts", {}),
        "outstanding_requirements": observation.get("outstanding_requirements", []),
        "action_history": observation.get("action_history", []),
        "last_feedback": observation.get("last_feedback", ""),
        "available_operations": observation.get("metadata", {}).get(
            "available_operations", []
        ),
    }
    return (
        "Based on the current ticket state below, choose the single best next "
        "action and return JSON only.\n\n"
        + json.dumps(context, indent=2)
    )


def _call_llm(observation: dict) -> dict:
    """Ask the LLM for the next action given the current observation."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(observation)},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    return json.loads(raw)


# ── main loop ──────────────────────────────────────────────────────────────

def run_task(task_id: str) -> Dict[str, Any]:
    """Run a single task and return the grader result."""

    rewards: List[float] = []
    step_count = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment
        reset_resp = _post("/reset", {"task_id": task_id, "seed": 7})
        observation = reset_resp.get("observation", reset_resp)

        done = False

        while not done and step_count < MAX_STEPS_PER_TASK:
            # Ask the LLM for the next action
            action = _call_llm(observation)
            action_str = action.get("operation", "unknown")

            # Execute the action
            step_resp = _post("/step", action)
            observation = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", observation.get("reward", 0.0))
            done = step_resp.get("done", observation.get("done", False))
            error = step_resp.get("last_action_error", None)
            step_count += 1

            rewards.append(reward)

            log_step(
                step=step_count,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

        # Grade the episode
        episode_export = observation.get("metadata", {}).get("episode_export", {})
        grader_resp = _post(
            "/grader",
            {"task_id": task_id, "episode": episode_export},
        )

        score = grader_resp.get("score", 0.0)
        passed = grader_resp.get("passed", False)
        success = passed

    except Exception:
        pass  # [END] is always emitted in the finally block

    finally:
        log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "passed": success,
        "steps_taken": step_count,
    }


def main() -> None:
    for task_id in TASK_IDS:
        run_task(task_id)


if __name__ == "__main__":
    main()
