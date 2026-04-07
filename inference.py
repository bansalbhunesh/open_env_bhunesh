"""
Inference script for the Support Ops OpenEnv environment.
===================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
from typing import List, Optional

from openai import OpenAI

from support_ops_env import SupportOpsAction, SupportOpsEnv

# ---------------------------------------------------------------------------
# Environment variables (mandatory per hackathon rules)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

BENCHMARK = "support_ops_env"
TASK_IDS: List[str] = [
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


# ── structured logging helpers (matches official format exactly) ───────────

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


# ── LLM helper ────────────────────────────────────────────────────────────

def get_action(llm_client: OpenAI, observation) -> dict:
    """Ask the LLM for the next action given the current observation."""
    obs_dict = observation.model_dump(mode="json") if hasattr(observation, "model_dump") else {}
    context = {
        "task_id": obs_dict.get("task_id", ""),
        "task_title": obs_dict.get("task_title", ""),
        "difficulty": obs_dict.get("difficulty", ""),
        "goal": obs_dict.get("goal", ""),
        "customer_message": obs_dict.get("customer_message", ""),
        "customer_profile": obs_dict.get("customer_profile", {}),
        "knowledge_base": obs_dict.get("knowledge_base", []),
        "ticket_snapshot": obs_dict.get("ticket_snapshot", {}),
        "revealed_customer_facts": obs_dict.get("revealed_customer_facts", {}),
        "outstanding_requirements": obs_dict.get("outstanding_requirements", []),
        "action_history": obs_dict.get("action_history", []),
        "last_feedback": obs_dict.get("last_feedback", ""),
        "available_operations": obs_dict.get("metadata", {}).get("available_operations", []),
    }
    user_prompt = (
        "Based on the current ticket state below, choose the single best next "
        "action and return JSON only.\n\n"
        + json.dumps(context, indent=2)
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

async def run_task(llm_client: OpenAI, env, task_id: str) -> dict:
    """Run a single task episode."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id, seed=7)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict = get_action(llm_client, observation)
            action_str = action_dict.get("operation", "unknown")
            action = SupportOpsAction.model_validate(action_dict)

            result = await env.step(action)
            observation = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Compute score from grader
        score = observation.model_dump(mode="json").get("metadata", {}).get(
            "episode_export", {}
        ).get("progress_score", sum(rewards) / max(len(rewards), 1))
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.75

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "passed": success, "steps": steps_taken}


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with SupportOpsEnv(base_url=ENV_SERVER_URL) as env:
        for task_id in TASK_IDS:
            await run_task(llm_client, env, task_id)


if __name__ == "__main__":
    asyncio.run(main())
