from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    instruction: str
    constraints: dict[str, Any] = field(default_factory=dict)
    source_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedCandidate:
    item_id: str
    title: str
    score_stage1: float
    score_stage2: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def final_score(self) -> float:
        return self.score_stage2 if self.score_stage2 > 0 else self.score_stage1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_score"] = self.final_score()
        return payload


@dataclass(frozen=True)
class RetrievedContext:
    task_id: str
    stage1_query: str
    candidates: tuple[RetrievedCandidate, ...]
    used_reranker: bool = False
    source_hint_respected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage1_query": self.stage1_query,
            "used_reranker": self.used_reranker,
            "source_hint_respected": self.source_hint_respected,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class PlanStep:
    title: str
    purpose: str
    expected_check: str
    risk: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningPlan:
    summary: str
    steps: tuple[PlanStep, ...]
    target_edits: tuple[str, ...]
    risks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "target_edits": list(self.target_edits),
            "risks": list(self.risks),
        }


@dataclass(frozen=True)
class OperatorProposal:
    operator_name: str
    arguments: dict[str, Any]
    confidence: float
    rationale: str
    source_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "arguments": dict(self.arguments),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True)
class SemanticCheck:
    passed: bool
    notes: tuple[str, ...] = ()
    repair_request: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "notes": list(self.notes),
            "repair_request": list(self.repair_request),
        }


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    structural_issues: tuple[dict[str, Any], ...] = ()
    hard_constraint_issues: tuple[dict[str, Any], ...] = ()
    dependency_issues: tuple[dict[str, Any], ...] = ()
    semantic_check: SemanticCheck | None = None

    @property
    def structural_ok(self) -> bool:
        return not self.structural_issues

    @property
    def hard_constraints_ok(self) -> bool:
        return not self.hard_constraint_issues

    @property
    def dependency_ok(self) -> bool:
        return not self.dependency_issues

    @property
    def violation_count(self) -> int:
        return len(self.structural_issues) + len(self.hard_constraint_issues) + len(self.dependency_issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "structural_ok": self.structural_ok,
            "hard_constraints_ok": self.hard_constraints_ok,
            "dependency_ok": self.dependency_ok,
            "violation_count": self.violation_count,
            "structural_issues": list(self.structural_issues),
            "hard_constraint_issues": list(self.hard_constraint_issues),
            "dependency_issues": list(self.dependency_issues),
            "semantic_check": self.semantic_check.to_dict() if self.semantic_check else None,
        }


@dataclass(frozen=True)
class CritiqueResult:
    approved: bool
    notes: tuple[str, ...]
    repair_proposals: tuple[OperatorProposal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "notes": list(self.notes),
            "repair_proposals": [proposal.to_dict() for proposal in self.repair_proposals],
        }


@dataclass(frozen=True)
class ModelCall:
    provider: str
    operation: str
    prompt_summary: str
    response_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningTrace:
    task: TaskSpec
    retrieved_context: RetrievedContext
    plan: ReasoningPlan | None
    proposals: tuple[OperatorProposal, ...]
    applied_actions: tuple[dict[str, Any], ...]
    rejected_actions: tuple[dict[str, Any], ...]
    verification: VerificationResult
    critique: CritiqueResult | None
    model_calls: tuple[ModelCall, ...]
    final_output: dict[str, Any]
    notes: tuple[str, ...] = ()

    def trace_completeness(self) -> float:
        checks = [
            bool(self.retrieved_context.candidates),
            self.plan is not None,
            bool(self.proposals),
            bool(self.applied_actions) or bool(self.rejected_actions),
            self.verification is not None,
            bool(self.model_calls),
            bool(self.final_output),
        ]
        return round(sum(1 for item in checks if item) / len(checks), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "retrieved_context": self.retrieved_context.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "applied_actions": list(self.applied_actions),
            "rejected_actions": list(self.rejected_actions),
            "verification": self.verification.to_dict(),
            "critique": self.critique.to_dict() if self.critique else None,
            "model_calls": [call.to_dict() for call in self.model_calls],
            "trace_completeness": self.trace_completeness(),
            "final_output": self.final_output,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SystemRun:
    system_name: str
    task: TaskSpec
    success: bool
    runtime_ms: float
    constraint_pass_rate: float
    minimal_edit_score: float
    model_call_count: int
    trace: ReasoningTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "task": self.task.to_dict(),
            "success": self.success,
            "runtime_ms": round(self.runtime_ms, 3),
            "constraint_pass_rate": self.constraint_pass_rate,
            "minimal_edit_score": self.minimal_edit_score,
            "model_call_count": self.model_call_count,
            "trace": self.trace.to_dict(),
        }
