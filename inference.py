#!/usr/bin/env python3
"""
Inference script for the Support Ops OpenEnv environment.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import time

# Force unbuffered stdout so the validator captures every line immediately
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (mandatory per hackathon rules)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

BENCHMARK = "support_ops_env"
TASK_IDS = [
    "password_reset_lockout",
    "duplicate_invoice_refund",
    "gdpr_export_incident",
]
MAX_STEPS = 8

SYSTEM_PROMPT = (
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

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── structured logging (matches official format exactly) ───────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = error if error else "null"
    d = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={d} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)


# ── HTTP helpers (synchronous, minimal, no WebSocket) ──────────────────────

def env_post(endpoint, payload=None):
    url = f"{ENV_SERVER_URL}{endpoint}"
    resp = requests.post(url, json=payload or {}, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ── LLM call ──────────────────────────────────────────────────────────────

def call_llm(observation):
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
        "available_operations": observation.get("metadata", {}).get("available_operations", []),
    }
    user_prompt = (
        "Based on the current ticket state below, choose the single best next "
        "action and return JSON only.\n\n" + json.dumps(context, indent=2)
    )
    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return {"operation": "resolve", "resolution_code": "guidance_provided",
                "resolution_summary": "Resolved with guidance."}


# ── main loop ──────────────────────────────────────────────────────────────

def run_task(task_id):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    # [START] is printed FIRST, before any network I/O
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = env_post("/reset", {"task_id": task_id, "seed": 7})
        observation = reset_resp.get("observation", reset_resp)

        done = False
        while not done and steps_taken < MAX_STEPS:
            action = call_llm(observation)
            action_str = action.get("operation", "unknown")

            step_resp = env_post("/step", action)
            observation = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", observation.get("reward", 0.0))
            done = step_resp.get("done", observation.get("done", False))
            error = step_resp.get("last_action_error", None)
            steps_taken += 1

            rewards.append(reward)
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

        # Grade the episode
        episode_export = observation.get("metadata", {}).get("episode_export", {})
        grader_resp = env_post("/grader", {"task_id": task_id, "episode": episode_export})
        score = grader_resp.get("score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = grader_resp.get("passed", False)

    except Exception:
        pass  # [END] is always emitted in finally

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    for task_id in TASK_IDS:
        run_task(task_id)


if __name__ == "__main__":
    main()
