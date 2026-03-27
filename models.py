# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the support operations environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionOperation(str, Enum):
    TRIAGE = "triage"
    REQUEST_CUSTOMER_INFO = "request_customer_info"
    ADD_INTERNAL_NOTE = "add_internal_note"
    SEND_REPLY = "send_reply"
    RESOLVE = "resolve"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketQueue(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    FINANCE_OPS = "finance_ops"
    PRIVACY_RESPONSE = "privacy_response"


class TicketCategory(str, Enum):
    ACCESS_MANAGEMENT = "access_management"
    BILLING_AND_REFUNDS = "billing_and_refunds"
    SECURITY_AND_PRIVACY = "security_and_privacy"


class TicketStatus(str, Enum):
    OPEN = "open"
    TRIAGED = "triaged"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    PENDING_INTERNAL = "pending_internal"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ResolutionCode(str, Enum):
    GUIDANCE_PROVIDED = "guidance_provided"
    REFUND_REVIEW_OPENED = "refund_review_opened"
    ESCALATED_TO_PRIVACY_RESPONSE = "escalated_to_privacy_response"


class TicketSnapshot(BaseModel):
    priority: Optional[TicketPriority] = None
    queue: Optional[TicketQueue] = None
    category: Optional[TicketCategory] = None
    status: TicketStatus = TicketStatus.OPEN
    tags: List[str] = Field(default_factory=list)
    resolution_code: Optional[ResolutionCode] = None
    resolution_summary: str = ""


class HistoryEvent(BaseModel):
    step: int
    operation: ActionOperation
    summary: str
    reward_delta: float = 0.0


class TaskDescriptor(BaseModel):
    task_id: str
    title: str
    difficulty: TaskDifficulty
    goal: str
    success_criteria: List[str]
    max_steps: int


class EpisodeExport(BaseModel):
    task_id: str
    task_title: str
    difficulty: TaskDifficulty
    steps_taken: int
    ticket_snapshot: TicketSnapshot
    revealed_customer_facts: Dict[str, str] = Field(default_factory=dict)
    info_requests_made: List[str] = Field(default_factory=list)
    internal_notes: List[str] = Field(default_factory=list)
    public_replies: List[str] = Field(default_factory=list)
    history: List[HistoryEvent] = Field(default_factory=list)
    progress_score: float = 0.0


class GraderCriterion(BaseModel):
    name: str
    score: float
    reason: str


class GraderRequest(BaseModel):
    task_id: Optional[str] = None
    episode: EpisodeExport


class GraderResponse(BaseModel):
    task_id: str
    title: str
    score: float
    passed: bool
    breakdown: List[GraderCriterion]
    missed_requirements: List[str] = Field(default_factory=list)


class BaselineTaskResult(BaseModel):
    task_id: str
    title: str
    score: float
    passed: bool
    steps_taken: int


class BaselineResponse(BaseModel):
    average_score: float
    tasks: List[BaselineTaskResult]


class TaskCatalogResponse(BaseModel):
    tasks: List[TaskDescriptor]


class SupportOpsAction(Action):
    """Single action model for ticket triage and resolution work."""

    operation: ActionOperation = Field(
        ...,
        description="Workflow operation to execute for this step.",
    )
    priority: Optional[TicketPriority] = Field(
        default=None,
        description="Priority to assign during triage.",
    )
    queue: Optional[TicketQueue] = Field(
        default=None,
        description="Queue to assign during triage.",
    )
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Category to assign during triage.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Normalized tags to attach to the ticket.",
    )
    requested_fields: List[str] = Field(
        default_factory=list,
        description="Customer fields to request when information is missing.",
    )
    internal_note: Optional[str] = Field(
        default=None,
        description="Internal analyst note for downstream teams.",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Customer-facing message.",
    )
    resolution_code: Optional[ResolutionCode] = Field(
        default=None,
        description="Resolution or escalation code used when closing the task.",
    )
    resolution_summary: Optional[str] = Field(
        default=None,
        description="Short summary explaining the chosen resolution.",
    )


class SupportOpsObservation(Observation):
    """Observation returned after each support operation step."""

    task_id: str = Field(default="", description="Current task identifier.")
    task_title: str = Field(default="", description="Human readable task title.")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Difficulty level for the current task.",
    )
    ticket_id: str = Field(default="", description="Deterministic ticket identifier.")
    goal: str = Field(default="", description="High-level task objective.")
    customer_message: str = Field(
        default="",
        description="Original customer message that must be handled.",
    )
    customer_profile: Dict[str, str] = Field(
        default_factory=dict,
        description="Structured customer and account context.",
    )
    knowledge_base: List[str] = Field(
        default_factory=list,
        description="Relevant internal support policy snippets.",
    )
    revealed_customer_facts: Dict[str, str] = Field(
        default_factory=dict,
        description="Facts revealed after requesting additional information.",
    )
    ticket_snapshot: TicketSnapshot = Field(
        default_factory=TicketSnapshot,
        description="Current structured state of the ticket.",
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Short summaries of prior actions taken in this episode.",
    )
    outstanding_requirements: List[str] = Field(
        default_factory=list,
        description="What remains to complete a strong resolution.",
    )
    last_feedback: str = Field(
        default="",
        description="Immediate environment feedback for the most recent action.",
    )


class SupportOpsState(State):
    """Serializable environment state for current support workflow."""

    task_id: str = ""
    task_title: str = ""
    difficulty: TaskDifficulty = TaskDifficulty.EASY
    ticket_id: str = ""
    goal: str = ""
    customer_message: str = ""
    customer_profile: Dict[str, str] = Field(default_factory=dict)
    knowledge_base: List[str] = Field(default_factory=list)
    ticket_snapshot: TicketSnapshot = Field(default_factory=TicketSnapshot)
    revealed_customer_facts: Dict[str, str] = Field(default_factory=dict)
    info_requests_made: List[str] = Field(default_factory=list)
    internal_notes: List[str] = Field(default_factory=list)
    public_replies: List[str] = Field(default_factory=list)
    progress_score: float = 0.0
    history: List[HistoryEvent] = Field(default_factory=list)
    completed_objectives: List[str] = Field(default_factory=list)
    remaining_objectives: List[str] = Field(default_factory=list)
    last_feedback: str = ""
