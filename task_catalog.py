"""Scenario catalog for the support operations environment."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .models import (
        ResolutionCode,
        TaskDifficulty,
        TaskDescriptor,
        TicketCategory,
        TicketPriority,
        TicketQueue,
    )
except ImportError:
    from models import (
        ResolutionCode,
        TaskDifficulty,
        TaskDescriptor,
        TicketCategory,
        TicketPriority,
        TicketQueue,
    )


@dataclass(frozen=True)
class ScenarioSpec:
    task_id: str
    title: str
    difficulty: TaskDifficulty
    goal: str
    customer_message: str
    customer_profile: dict[str, str]
    knowledge_base: list[str]
    required_customer_fields: dict[str, str]
    success_criteria: list[str]
    gold_priority: TicketPriority
    gold_queue: TicketQueue
    gold_category: TicketCategory
    gold_tags: list[str]
    internal_note_keywords: list[str]
    public_reply_keywords: list[str]
    public_reply_forbidden: list[str]
    resolution_code: ResolutionCode
    resolution_summary_keywords: list[str]
    max_steps: int
    target_steps: int

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            title=self.title,
            difficulty=self.difficulty,
            goal=self.goal,
            success_criteria=self.success_criteria,
            max_steps=self.max_steps,
        )


SCENARIOS: dict[str, ScenarioSpec] = {
    "password_reset_lockout": ScenarioSpec(
        task_id="password_reset_lockout",
        title="Password reset for locked-out user",
        difficulty=TaskDifficulty.EASY,
        goal=(
            "Triage the access issue, capture the right operational context, "
            "reply with practical recovery guidance, and close the ticket cleanly."
        ),
        customer_message=(
            "Hi support, I am locked out of Nimbus after rotating laptops. "
            "Password reset emails arrive, but the link shows as expired right away. "
            "I need access before our 3pm onboarding call. Workspace: Acme Design. "
            "Account email: alyssa@acme.design. We are on the Growth plan."
        ),
        customer_profile={
            "customer_name": "Alyssa Tran",
            "company": "Acme Design",
            "plan_tier": "growth",
            "region": "US",
            "account_email": "alyssa@acme.design",
        },
        knowledge_base=[
            "Expired reset links are typically caused by cached older email links or link prefetching.",
            "Access issues for Growth accounts stay in customer_support unless there is a platform outage.",
            "A complete reply should include immediate recovery steps and confirm follow-up if the issue persists.",
        ],
        required_customer_fields={},
        success_criteria=[
            "Classify the ticket as access management and route it to customer support.",
            "Mark the issue high priority because the user is blocked before a scheduled onboarding call.",
            "Respond with concrete recovery steps for an expired password reset link.",
        ],
        gold_priority=TicketPriority.HIGH,
        gold_queue=TicketQueue.CUSTOMER_SUPPORT,
        gold_category=TicketCategory.ACCESS_MANAGEMENT,
        gold_tags=["password-reset", "login", "growth-plan"],
        internal_note_keywords=["expired reset link", "growth plan", "onboarding call"],
        public_reply_keywords=["reset link", "browser", "support", "onboarding"],
        public_reply_forbidden=["refund", "invoice", "security breach"],
        resolution_code=ResolutionCode.GUIDANCE_PROVIDED,
        resolution_summary_keywords=["guided", "password reset", "access restored"],
        max_steps=5,
        target_steps=4,
    ),
    "duplicate_invoice_refund": ScenarioSpec(
        task_id="duplicate_invoice_refund",
        title="Duplicate invoice after annual upgrade",
        difficulty=TaskDifficulty.MEDIUM,
        goal=(
            "Handle a billing dispute by routing it correctly, requesting the missing "
            "finance evidence, communicating the next steps, and opening refund review."
        ),
        customer_message=(
            "We upgraded from monthly to annual yesterday and were charged both invoices. "
            "Finance needs this fixed before close. Company: Northwind Logistics. "
            "Account owner: meera@northwindlogistics.com. Plan: Business. "
            "I only see one invoice in the admin panel."
        ),
        customer_profile={
            "customer_name": "Meera Nair",
            "company": "Northwind Logistics",
            "plan_tier": "business",
            "region": "DE",
            "account_email": "meera@northwindlogistics.com",
        },
        knowledge_base=[
            "Duplicate charge investigations require invoice evidence before a refund review can be opened.",
            "Billing disputes for Business accounts route to finance_ops, not general support.",
            "The customer-facing reply should set an expectation that finance reviews duplicate charge cases within 24 hours.",
        ],
        required_customer_fields={
            "invoice_id": "INV-44821",
            "card_last4": "4242",
            "billing_country": "DE",
        },
        success_criteria=[
            "Route the ticket to finance operations with billing-and-refunds categorization.",
            "Collect the invoice id, card last 4, and billing country before final resolution.",
            "Explain that a refund review is being opened and provide a realistic turnaround.",
        ],
        gold_priority=TicketPriority.HIGH,
        gold_queue=TicketQueue.FINANCE_OPS,
        gold_category=TicketCategory.BILLING_AND_REFUNDS,
        gold_tags=["duplicate-charge", "invoice", "annual-upgrade"],
        internal_note_keywords=["double charge", "annual upgrade", "refund review"],
        public_reply_keywords=["invoice", "refund", "finance", "24 hours"],
        public_reply_forbidden=["security", "password reset"],
        resolution_code=ResolutionCode.REFUND_REVIEW_OPENED,
        resolution_summary_keywords=["refund review", "duplicate charge", "annual plan"],
        max_steps=6,
        target_steps=5,
    ),
    "gdpr_export_incident": ScenarioSpec(
        task_id="gdpr_export_incident",
        title="EU privacy request after mistaken employee export",
        difficulty=TaskDifficulty.HARD,
        goal=(
            "Contain a potential privacy incident for an enterprise tenant by escalating "
            "to the privacy queue, gathering the minimum verification data, and sending "
            "a careful customer response with compliance-aware language."
        ),
        customer_message=(
            "I am the IT admin for Helix Bio EU. A contractor account downloaded an "
            "employee export that included terminated staff. We need urgent containment "
            "guidance, deletion guidance, and an audit trail for legal. "
            "This may be report people-export-77. Our contract says enterprise support "
            "with a 4-hour response SLA."
        ),
        customer_profile={
            "customer_name": "Sofia Keller",
            "company": "Helix Bio EU",
            "plan_tier": "enterprise",
            "region": "EU",
            "account_email": "sofia.keller@helixbio.eu",
        },
        knowledge_base=[
            "Potential privacy incidents involving EU employee data must route to privacy_response immediately.",
            "Analysts should document containment, legal escalation, and the 72-hour notification window in internal notes.",
            "Customer replies should acknowledge privacy and security review, share the next checkpoint, and avoid promising deletion before investigation.",
        ],
        required_customer_fields={
            "tenant_id": "helix-bio-eu",
            "legal_contact": "dpo@helixbio.eu",
            "affected_report_id": "people-export-77",
        },
        success_criteria=[
            "Escalate to the privacy response queue with urgent priority and security/privacy categorization.",
            "Collect tenant, legal contact, and affected report details before closing out the workflow.",
            "Document containment and legal considerations internally and send a careful public reply without overpromising.",
        ],
        gold_priority=TicketPriority.URGENT,
        gold_queue=TicketQueue.PRIVACY_RESPONSE,
        gold_category=TicketCategory.SECURITY_AND_PRIVACY,
        gold_tags=["gdpr", "security-incident", "audit-log"],
        internal_note_keywords=["72-hour", "containment", "legal", "security"],
        public_reply_keywords=["privacy", "security", "audit", "timeline"],
        public_reply_forbidden=["guarantee deletion today", "no breach"],
        resolution_code=ResolutionCode.ESCALATED_TO_PRIVACY_RESPONSE,
        resolution_summary_keywords=["privacy response", "containment", "audit trail"],
        max_steps=7,
        target_steps=5,
    ),
}


ORDERED_TASK_IDS = [
    "password_reset_lockout",
    "duplicate_invoice_refund",
    "gdpr_export_incident",
]


def get_scenario(task_id: str) -> ScenarioSpec:
    return SCENARIOS[task_id]


def list_task_descriptors() -> list[TaskDescriptor]:
    return [SCENARIOS[task_id].descriptor() for task_id in ORDERED_TASK_IDS]
