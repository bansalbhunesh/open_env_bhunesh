---
title: Support Ops Environment Server
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - support-ops
  - rl
---

# Support Ops Environment

`support_ops_env` is a real-world OpenEnv benchmark for SaaS support operations. The agent acts like a frontline support analyst handling live customer tickets: triaging, gathering missing evidence, writing internal notes, sending customer replies, and closing or escalating the case.

This environment is designed for RL or agent-evaluation workflows that need:

- realistic ticket routing and operational judgment
- partial-progress rewards instead of sparse terminal-only signals
- deterministic graders with score outputs in the `0.0` to `1.0` range
- reproducible baseline rollouts across easy, medium, and hard tasks

## Included Tasks

The environment ships with three deterministic tasks:

1. `password_reset_lockout` (`easy`): resolve a blocked user before an onboarding call.
2. `duplicate_invoice_refund` (`medium`): gather billing evidence and open refund review.
3. `gdpr_export_incident` (`hard`): escalate a privacy/security incident for an enterprise EU tenant.

Use `GET /tasks` to inspect the task catalog and success criteria.

## OpenEnv API

The environment implements the standard OpenEnv interface:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

It also exposes the extra endpoints required by the submission validator:

- `GET /tasks`
- `POST /grader`
- `GET|POST /baseline`

`/tasks` returns both the task list and the typed `SupportOpsAction` JSON schema.

## Action Space

The typed action model is `SupportOpsAction`.

Fields:

- `operation`: one of `triage`, `request_customer_info`, `add_internal_note`, `send_reply`, `resolve`
- `priority`, `queue`, `category`: structured ticket triage fields
- `tags`: normalized support tags to attach to the ticket
- `requested_fields`: customer evidence to collect for the current workflow
- `internal_note`: internal-only summary for downstream teams
- `response_text`: customer-facing reply
- `resolution_code`, `resolution_summary`: final resolution or escalation output

## Observation Space

The typed observation model is `SupportOpsObservation`.

Core fields:

- `task_id`, `task_title`, `difficulty`
- `ticket_id`, `goal`
- `customer_message`, `customer_profile`
- `knowledge_base`: short policy/runbook snippets relevant to the ticket
- `revealed_customer_facts`: additional facts unlocked after requesting info
- `ticket_snapshot`: current queue, category, priority, status, tags, and resolution state
- `action_history`: short summaries of prior agent actions
- `outstanding_requirements`: what the grader still expects
- `last_feedback`: immediate reward-shaping feedback
- standard OpenEnv fields: `reward`, `done`, `metadata`

The `metadata` payload includes `episode_export`, which can be sent directly to `/grader`.

## Reward Design

Rewards are shaped as the delta between the current workflow score and the previous step score, plus small operation penalties/bonuses:

- positive reward for improving routing, tagging, note quality, reply quality, and resolution quality
- extra reward when a useful customer-info request reveals hidden evidence
- small per-step penalty to discourage dithering
- stronger penalties for incomplete or invalid actions
- final bonus when the workflow is completed with a passing grader score

This produces dense, meaningful reward instead of a single sparse terminal signal.

## Graders

`POST /grader` accepts an `EpisodeExport` and returns a deterministic `0.0` to `1.0` score.

The grader evaluates:

- routing correctness
- tag coverage
- information gathering
- internal note quality
- customer reply quality
- resolution quality
- efficiency and duplicate actions

The response includes a breakdown and missed requirements list.

## Quick Start

### 1. Install dependencies

If you have `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

### 2. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Inspect the task catalog

```bash
curl http://localhost:8000/tasks
```

### 4. Run the baseline script (OpenAI mode)

```bash
set OPENAI_API_KEY=your_key_here
python -m support_ops_env.baseline_inference --base-url http://localhost:8000 --mode openai --model gpt-4o-mini
```

Deterministic fallback mode (no model API calls):

```bash
python -m support_ops_env.baseline_inference --base-url http://localhost:8000 --mode deterministic
```

### 5. Run the built-in baseline endpoint

```bash
curl http://localhost:8000/baseline
```

You can also request OpenAI baseline mode through the endpoint:

```bash
curl "http://localhost:8000/baseline?mode=openai"
```

## Example Client Usage

```python
from support_ops_env import SupportOpsAction, SupportOpsEnv
from support_ops_env.models import (
    ActionOperation,
    ResolutionCode,
    TicketCategory,
    TicketPriority,
    TicketQueue,
)

client = SupportOpsEnv(base_url="http://localhost:8000").sync()

with client:
    result = client.reset(task_id="password_reset_lockout", seed=7)

    result = client.step(
        SupportOpsAction(
            operation=ActionOperation.TRIAGE,
            priority=TicketPriority.HIGH,
            queue=TicketQueue.CUSTOMER_SUPPORT,
            category=TicketCategory.ACCESS_MANAGEMENT,
            tags=["password-reset", "login", "growth-plan"],
        )
    )

    result = client.step(
        SupportOpsAction(
            operation=ActionOperation.SEND_REPLY,
            response_text=(
                "Please open the latest reset link in a fresh browser session. "
                "If the onboarding deadline is still at risk, reply here and support "
                "will extend live help."
            ),
        )
    )

    result = client.step(
        SupportOpsAction(
            operation=ActionOperation.RESOLVE,
            resolution_code=ResolutionCode.GUIDANCE_PROVIDED,
            resolution_summary="Guided customer through password reset recovery.",
        )
    )

    print(result.reward, result.done)
```

## Docker

Build locally:

```bash
docker build -t support_ops_env-env:latest -f server/Dockerfile .
```

Run locally:

```bash
docker run -p 8000:8000 support_ops_env-env:latest
```

## Hugging Face Spaces

This environment is configured as a Docker Space through `openenv.yaml`.

Deploy with the OpenEnv CLI:

```bash
openenv push
```

Or specify a target repository:

```bash
openenv push --repo-id <namespace>/support-ops-env
```

After deployment, the Space serves:

- `/web` for the interactive UI
- `/docs` for OpenAPI docs
- `/tasks`, `/grader`, `/baseline` for evaluation tooling

## Validation

Recommended local checks:

```bash
python -m support_ops_env.baseline_inference --base-url http://localhost:8000 --mode deterministic
openenv validate --verbose
```

## Fault Injection

For safe resilience testing, the environment includes an opt-in debug fault mode that is off by default.

Supported environment variables:

- `DEBUG_FAULT_INJECTION=latency`
- `DEBUG_FAULT_INJECTION=transient_error`
- `DEBUG_FAULT_LATENCY_MS=750`
- `DEBUG_FAULT_STEP=2`

Examples:

```bash
DEBUG_FAULT_INJECTION=latency DEBUG_FAULT_LATENCY_MS=1200 uvicorn server.app:app --host 0.0.0.0 --port 8000
```

```bash
DEBUG_FAULT_INJECTION=transient_error DEBUG_FAULT_STEP=3 uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Behavior:

- `latency` adds an artificial delay before every `step()`
- `transient_error` raises a single clearly labeled runtime error on the configured step, then resumes normal behavior after reset

This mode is intended for local debugging and client retry testing. Leave it unset for normal training, validation, and deployment.

## Project Layout

```text
support_ops_env/
  __init__.py
  baseline_inference.py
  client.py
  evaluation.py
  models.py
  openenv.yaml
  pyproject.toml
  README.md
  task_catalog.py
  server/
    app.py
    support_ops_env_environment.py
    Dockerfile
```
