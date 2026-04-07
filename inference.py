"""
Inference script for the Support Ops OpenEnv environment.

Uses the OpenAI Client with environment variables:
    API_BASE_URL  – LLM API endpoint
    MODEL_NAME    – model identifier
    HF_TOKEN      – Hugging Face / API key

Emits structured stdout logs in [START], [STEP], [END] format.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

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

    # Reset the environment
    reset_resp = _post("/reset", {"task_id": task_id, "seed": 7})
    observation = reset_resp.get("observation", reset_resp)

    # ── [START] ──
    print(f"[START] task={task_id}", flush=True)

    done = False
    step_count = 0
    reward = 0.0

    while not done and step_count < MAX_STEPS_PER_TASK:
        # Ask the LLM for the next action
        action = _call_llm(observation)

        # Execute the action
        step_resp = _post("/step", action)
        observation = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", observation.get("reward", 0.0))
        done = step_resp.get("done", observation.get("done", False))
        step_count += 1

        # ── [STEP] ──
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    # Grade the episode
    episode_export = observation.get("metadata", {}).get("episode_export", {})
    grader_resp = _post(
        "/grader",
        {"task_id": task_id, "episode": episode_export},
    )

    score = grader_resp.get("score", 0.0)
    passed = grader_resp.get("passed", False)

    # ── [END] ──
    print(f"[END] task={task_id} score={score} steps={step_count}", flush=True)

    return {
        "task_id": task_id,
        "score": score,
        "passed": passed,
        "steps_taken": step_count,
    }


def main() -> None:
    results: List[Dict[str, Any]] = []

    for task_id in TASK_IDS:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as exc:
            # Silence internal generic errors that would corrupt stdout parsing
            # Just print the cleanly expected [END]
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            results.append(
                {
                    "task_id": task_id,
                    "score": 0.0,
                    "passed": False,
                    "steps_taken": 0,
                    "error": str(exc),
                }
            )

if __name__ == "__main__":
    main()
