# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Support operations environment implementation."""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..evaluation import grade_episode_export, remaining_objectives
    from ..models import (
        ActionOperation,
        EpisodeExport,
        HistoryEvent,
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsState,
        TaskDifficulty,
        TicketSnapshot,
        TicketStatus,
    )
    from ..task_catalog import ORDERED_TASK_IDS, get_scenario
except ImportError:
    from evaluation import grade_episode_export, remaining_objectives
    from models import (
        ActionOperation,
        EpisodeExport,
        HistoryEvent,
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsState,
        TaskDifficulty,
        TicketSnapshot,
        TicketStatus,
    )
    from task_catalog import ORDERED_TASK_IDS, get_scenario


class SupportOpsEnvironment(Environment):
    """Ticket triage and support workflow environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._rng = random.Random()
        self._state = SupportOpsState(episode_id=str(uuid4()), step_count=0)
        self._task_id = ORDERED_TASK_IDS[0]
        self._required_customer_fields: dict[str, str] = {}
        self._resolved = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        difficulty: str | None = None,
        **_: object,
    ) -> SupportOpsObservation:
        if seed is not None:
            self._rng.seed(seed)

        if task_id is None and difficulty is not None:
            requested_difficulty = TaskDifficulty(difficulty)
            matching = [
                candidate
                for candidate in ORDERED_TASK_IDS
                if get_scenario(candidate).difficulty == requested_difficulty
            ]
            task_id = matching[0]

        self._task_id = task_id or ORDERED_TASK_IDS[0]
        scenario = get_scenario(self._task_id)
        ticket_id = f"{scenario.task_id}-{self._rng.randint(1000, 9999)}"
        episode_identifier = episode_id or str(uuid4())
        self._required_customer_fields = dict(scenario.required_customer_fields)
        self._resolved = False

        self._state = SupportOpsState(
            episode_id=episode_identifier,
            step_count=0,
            task_id=scenario.task_id,
            task_title=scenario.title,
            difficulty=scenario.difficulty,
            ticket_id=ticket_id,
            goal=scenario.goal,
            customer_message=scenario.customer_message,
            customer_profile=scenario.customer_profile,
            knowledge_base=scenario.knowledge_base,
            ticket_snapshot=TicketSnapshot(status=TicketStatus.OPEN),
            remaining_objectives=list(scenario.success_criteria),
            last_feedback=(
                "Start by triaging the ticket. Request additional facts only when needed."
            ),
        )
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: SupportOpsAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> SupportOpsObservation:
        del timeout_s

        if self._resolved:
            return self._build_observation(
                reward=-0.1,
                done=True,
                feedback="Episode already completed. Reset to start a new task.",
            )

        self._state.step_count += 1
        previous_progress = self._state.progress_score
        reward_adjustment = -0.02
        feedback_parts: list[str] = []
        history_summary = ""

        if action.tags:
            current_tags = {_normalize_tag(tag) for tag in self._state.ticket_snapshot.tags}
            current_tags.update(_normalize_tag(tag) for tag in action.tags if tag.strip())
            self._state.ticket_snapshot.tags = sorted(current_tags)

        if action.operation == ActionOperation.TRIAGE:
            missing_fields = [
                field
                for field in ("priority", "queue", "category")
                if getattr(action, field) is None
            ]
            if missing_fields:
                reward_adjustment -= 0.08
                feedback_parts.append(
                    "Triage actions must include priority, queue, and category."
                )
                history_summary = "Attempted incomplete triage."
            else:
                self._state.ticket_snapshot.priority = action.priority
                self._state.ticket_snapshot.queue = action.queue
                self._state.ticket_snapshot.category = action.category
                self._state.ticket_snapshot.status = TicketStatus.TRIAGED
                feedback_parts.append("Structured triage fields updated.")
                history_summary = (
                    f"Triaged to {action.queue.value} with {action.priority.value} priority."
                )

        elif action.operation == ActionOperation.REQUEST_CUSTOMER_INFO:
            requested_fields = [
                field.strip().lower()
                for field in action.requested_fields
                if field.strip()
            ]
            if not requested_fields:
                reward_adjustment -= 0.08
                feedback_parts.append(
                    "Request-customer-info actions must specify at least one field."
                )
                history_summary = "Requested customer info without specifying fields."
            else:
                newly_revealed = {}
                for field in requested_fields:
                    if field in self._required_customer_fields:
                        newly_revealed[field] = self._required_customer_fields[field]
                        self._state.revealed_customer_facts[field] = self._required_customer_fields[field]
                self._state.info_requests_made.extend(requested_fields)
                self._state.ticket_snapshot.status = TicketStatus.WAITING_ON_CUSTOMER
                if newly_revealed:
                    reward_adjustment += 0.05
                    feedback_parts.append(
                        "Customer responded with: "
                        + ", ".join(f"{key}={value}" for key, value in newly_revealed.items())
                    )
                    self._state.ticket_snapshot.status = TicketStatus.PENDING_INTERNAL
                    history_summary = (
                        "Requested additional customer evidence and received a follow-up."
                    )
                else:
                    reward_adjustment -= 0.04
                    feedback_parts.append(
                        "Those fields do not unlock any new information for this scenario."
                    )
                    history_summary = "Requested irrelevant customer information."

        elif action.operation == ActionOperation.ADD_INTERNAL_NOTE:
            if not action.internal_note or not action.internal_note.strip():
                reward_adjustment -= 0.08
                feedback_parts.append("Internal-note actions must include note text.")
                history_summary = "Submitted an empty internal note."
            else:
                note = action.internal_note.strip()
                self._state.internal_notes.append(note)
                self._state.ticket_snapshot.status = TicketStatus.PENDING_INTERNAL
                feedback_parts.append("Internal note recorded for downstream teams.")
                history_summary = "Recorded an internal support note."

        elif action.operation == ActionOperation.SEND_REPLY:
            if not action.response_text or not action.response_text.strip():
                reward_adjustment -= 0.08
                feedback_parts.append("Send-reply actions must include customer-facing text.")
                history_summary = "Attempted to send an empty customer reply."
            else:
                reply = action.response_text.strip()
                self._state.public_replies.append(reply)
                self._state.ticket_snapshot.status = TicketStatus.PENDING_INTERNAL
                feedback_parts.append("Customer reply drafted and stored.")
                history_summary = "Sent a customer-facing reply."

        elif action.operation == ActionOperation.RESOLVE:
            if not action.resolution_code or not action.resolution_summary:
                reward_adjustment -= 0.08
                feedback_parts.append(
                    "Resolve actions need both a resolution code and a resolution summary."
                )
                history_summary = "Attempted to resolve without a complete summary."
            else:
                self._state.ticket_snapshot.resolution_code = action.resolution_code
                self._state.ticket_snapshot.resolution_summary = action.resolution_summary.strip()
                if action.resolution_code.value.startswith("escalated"):
                    self._state.ticket_snapshot.status = TicketStatus.ESCALATED
                else:
                    self._state.ticket_snapshot.status = TicketStatus.RESOLVED
                self._resolved = True
                feedback_parts.append("Workflow marked complete.")
                history_summary = f"Resolved ticket with {action.resolution_code.value}."

        self._state.last_feedback = " ".join(feedback_parts) or "No material change."
        self._state.history.append(
            HistoryEvent(
                step=self._state.step_count,
                operation=action.operation,
                summary=history_summary or action.operation.value,
                reward_delta=0.0,
            )
        )

        episode = self._episode_export()
        grade = grade_episode_export(episode)
        self._state.progress_score = grade.score
        self._state.remaining_objectives = remaining_objectives(episode)
        self._state.completed_objectives = [
            criterion.name for criterion in grade.breakdown if criterion.score >= 0.95
        ]

        reward = round((grade.score - previous_progress) + reward_adjustment, 4)
        done = self._resolved or self._state.step_count >= get_scenario(self._task_id).max_steps
        if done:
            self._resolved = True
            if grade.passed:
                reward = round(reward + 0.1, 4)
            else:
                reward = round(reward - 0.05, 4)

        self._state.history[-1].reward_delta = reward

        return self._build_observation(
            reward=reward,
            done=done,
            feedback=self._state.last_feedback,
        )

    @property
    def state(self) -> SupportOpsState:
        return self._state

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.name = "SupportOpsEnvironment"
        metadata.description = (
            "A real-world support operations benchmark with ticket triage, "
            "customer follow-up, and deterministic graders."
        )
        return metadata

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback: str | None = None,
    ) -> SupportOpsObservation:
        scenario = get_scenario(self._task_id)
        episode = self._episode_export()
        grade = grade_episode_export(episode, scenario)

        return SupportOpsObservation(
            task_id=scenario.task_id,
            task_title=scenario.title,
            difficulty=scenario.difficulty,
            ticket_id=self._state.ticket_id,
            goal=scenario.goal,
            customer_message=scenario.customer_message,
            customer_profile=scenario.customer_profile,
            knowledge_base=scenario.knowledge_base,
            revealed_customer_facts=dict(self._state.revealed_customer_facts),
            ticket_snapshot=self._state.ticket_snapshot,
            action_history=[event.summary for event in self._state.history[-5:]],
            outstanding_requirements=list(self._state.remaining_objectives),
            last_feedback=feedback or self._state.last_feedback,
            reward=reward,
            done=done,
            metadata={
                "score_preview": grade.score,
                "grader_breakdown": [item.model_dump() for item in grade.breakdown],
                "episode_export": episode.model_dump(mode="json"),
                "available_operations": [operation.value for operation in ActionOperation],
            },
        )

    def _episode_export(self) -> EpisodeExport:
        return EpisodeExport(
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            difficulty=self._state.difficulty,
            steps_taken=self._state.step_count,
            ticket_snapshot=self._state.ticket_snapshot,
            revealed_customer_facts=dict(self._state.revealed_customer_facts),
            info_requests_made=list(self._state.info_requests_made),
            internal_notes=list(self._state.internal_notes),
            public_replies=list(self._state.public_replies),
            history=list(self._state.history),
            progress_score=self._state.progress_score,
        )


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace("_", "-")
