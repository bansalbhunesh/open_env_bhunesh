"""Shared grading and baseline helpers for support operations tasks."""

from __future__ import annotations

from statistics import mean

try:
    from .models import (
        ActionOperation,
        BaselineResponse,
        BaselineTaskResult,
        EpisodeExport,
        GraderCriterion,
        GraderResponse,
        ResolutionCode,
        SupportOpsAction,
    )
    from .task_catalog import ORDERED_TASK_IDS, ScenarioSpec, get_scenario
except ImportError:
    from models import (
        ActionOperation,
        BaselineResponse,
        BaselineTaskResult,
        EpisodeExport,
        GraderCriterion,
        GraderResponse,
        ResolutionCode,
        SupportOpsAction,
    )
    from task_catalog import ORDERED_TASK_IDS, ScenarioSpec, get_scenario


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace("_", "-")


def _coverage(items: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    normalized_items = {_normalize_token(item) for item in items if item.strip()}
    normalized_expected = {_normalize_token(item) for item in expected}
    return len(normalized_items & normalized_expected) / len(normalized_expected)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    haystack = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in haystack) / len(keywords)


def _forbidden_penalty(text: str, forbidden: list[str]) -> float:
    haystack = text.lower()
    if any(keyword.lower() in haystack for keyword in forbidden):
        return 0.4
    return 0.0


def _duplicate_operation_penalty(episode: EpisodeExport) -> float:
    seen: set[tuple[str, str]] = set()
    penalty = 0.0
    for event in episode.history:
        key = (event.operation.value, event.summary.strip().lower())
        if key in seen:
            penalty += 0.05
        else:
            seen.add(key)
    return min(penalty, 0.2)


def grade_episode_export(
    episode: EpisodeExport,
    scenario: ScenarioSpec | None = None,
) -> GraderResponse:
    scenario = scenario or get_scenario(episode.task_id)
    snapshot = episode.ticket_snapshot

    routing_score = mean(
        [
            1.0 if snapshot.category == scenario.gold_category else 0.0,
            1.0 if snapshot.queue == scenario.gold_queue else 0.0,
            1.0 if snapshot.priority == scenario.gold_priority else 0.0,
        ]
    )
    tag_score = _coverage(snapshot.tags, scenario.gold_tags)

    if scenario.required_customer_fields:
        revealed_fields = [
            field
            for field, value in scenario.required_customer_fields.items()
            if episode.revealed_customer_facts.get(field) == value
        ]
        info_score = len(revealed_fields) / len(scenario.required_customer_fields)
    else:
        info_score = 1.0

    note_text = "\n".join(episode.internal_notes)
    note_score = _keyword_coverage(note_text, scenario.internal_note_keywords)

    reply_text = episode.public_replies[-1] if episode.public_replies else ""
    reply_score = max(
        0.0,
        _keyword_coverage(reply_text, scenario.public_reply_keywords)
        - _forbidden_penalty(reply_text, scenario.public_reply_forbidden),
    )

    summary_text = snapshot.resolution_summary or ""
    resolution_score = mean(
        [
            1.0 if snapshot.resolution_code == scenario.resolution_code else 0.0,
            _keyword_coverage(summary_text, scenario.resolution_summary_keywords),
        ]
    )

    step_overrun = max(0, episode.steps_taken - scenario.target_steps)
    efficiency_score = max(
        0.0,
        1.0 - (step_overrun * 0.12) - _duplicate_operation_penalty(episode),
    )

    weighted = [
        (routing_score, 0.25),
        (tag_score, 0.10),
        (info_score, 0.15),
        (note_score, 0.15),
        (reply_score, 0.15),
        (resolution_score, 0.15),
        (efficiency_score, 0.05),
    ]
    score = round(sum(component * weight for component, weight in weighted), 4)

    breakdown = [
        GraderCriterion(
            name="routing",
            score=round(routing_score, 4),
            reason=(
                f"Expected {scenario.gold_category.value}, {scenario.gold_queue.value}, "
                f"{scenario.gold_priority.value}."
            ),
        ),
        GraderCriterion(
            name="tagging",
            score=round(tag_score, 4),
            reason=f"Expected tags aligned with {', '.join(scenario.gold_tags)}.",
        ),
        GraderCriterion(
            name="info_gathering",
            score=round(info_score, 4),
            reason=(
                "Collected required customer fields."
                if scenario.required_customer_fields
                else "No extra customer information was required."
            ),
        ),
        GraderCriterion(
            name="internal_note_quality",
            score=round(note_score, 4),
            reason="Internal notes should capture the critical operational context.",
        ),
        GraderCriterion(
            name="customer_reply_quality",
            score=round(reply_score, 4),
            reason="Customer reply should be useful, specific, and policy-safe.",
        ),
        GraderCriterion(
            name="resolution",
            score=round(resolution_score, 4),
            reason="Resolution code and summary should match the desired workflow outcome.",
        ),
        GraderCriterion(
            name="efficiency",
            score=round(efficiency_score, 4),
            reason="Shorter, non-redundant workflows score better.",
        ),
    ]

    missed_requirements: list[str] = []
    if routing_score < 1.0:
        missed_requirements.append("Ticket routing or priority is still incorrect.")
    if info_score < 1.0:
        missing = [
            field
            for field in scenario.required_customer_fields
            if field not in episode.revealed_customer_facts
        ]
        missed_requirements.append(
            f"Missing customer facts: {', '.join(sorted(missing))}."
        )
    if note_score < 0.75:
        missed_requirements.append("Internal note is missing key operational context.")
    if reply_score < 0.75:
        missed_requirements.append("Customer reply is incomplete or overpromises.")
    if resolution_score < 1.0:
        missed_requirements.append("Resolution code or summary does not match the task.")

    return GraderResponse(
        task_id=scenario.task_id,
        title=scenario.title,
        score=score,
        passed=score >= 0.75,
        breakdown=breakdown,
        missed_requirements=missed_requirements,
    )


def remaining_objectives(episode: EpisodeExport) -> list[str]:
    grade = grade_episode_export(episode)
    return grade.missed_requirements


def baseline_plan_for_task(task_id: str) -> list[SupportOpsAction]:
    scenario = get_scenario(task_id)
    plan = [
        SupportOpsAction(
            operation=ActionOperation.TRIAGE,
            priority=scenario.gold_priority,
            queue=scenario.gold_queue,
            category=scenario.gold_category,
            tags=scenario.gold_tags,
        )
    ]

    if scenario.required_customer_fields:
        plan.append(
            SupportOpsAction(
                operation=ActionOperation.REQUEST_CUSTOMER_INFO,
                requested_fields=list(scenario.required_customer_fields.keys()),
            )
        )

    plan.append(
        SupportOpsAction(
            operation=ActionOperation.ADD_INTERNAL_NOTE,
            internal_note=(
                "Escalation note: "
                + "; ".join(scenario.internal_note_keywords)
            ),
        )
    )
    plan.append(
        SupportOpsAction(
            operation=ActionOperation.SEND_REPLY,
            response_text=(
                "We are handling this through the appropriate team. "
                + " ".join(
                    f"We will cover {keyword}."
                    for keyword in scenario.public_reply_keywords
                )
            ),
        )
    )
    plan.append(
        SupportOpsAction(
            operation=ActionOperation.RESOLVE,
            resolution_code=scenario.resolution_code,
            resolution_summary=(
                "Handled by workflow: "
                + ", ".join(scenario.resolution_summary_keywords)
            ),
        )
    )
    return plan


def run_in_process_baseline(env_factory: type) -> BaselineResponse:
    results: list[BaselineTaskResult] = []

    for task_id in ORDERED_TASK_IDS:
        env = env_factory()
        env.reset(seed=7, task_id=task_id)
        final_observation = None

        for action in baseline_plan_for_task(task_id):
            final_observation = env.step(action)
            if final_observation.done:
                break

        if final_observation is None:
            raise RuntimeError("Baseline did not execute any actions.")

        episode = EpisodeExport.model_validate(
            final_observation.metadata["episode_export"]
        )
        grade = grade_episode_export(episode)
        results.append(
            BaselineTaskResult(
                task_id=task_id,
                title=grade.title,
                score=grade.score,
                passed=grade.passed,
                steps_taken=episode.steps_taken,
            )
        )
        env.close()

    average_score = round(mean(result.score for result in results), 4)
    return BaselineResponse(average_score=average_score, tasks=results)


def default_resolution_for_operation(operation: ActionOperation) -> ResolutionCode | None:
    if operation == ActionOperation.RESOLVE:
        return ResolutionCode.GUIDANCE_PROVIDED
    return None
