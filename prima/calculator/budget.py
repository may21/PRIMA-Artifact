"""PRIMA Calculator adapter.

The paper names this module Calculator. The earlier scaffold used
`prima.allocator.planner`; this adapter preserves that implementation while
exposing the paper-facing module name.
"""

from __future__ import annotations

from dataclasses import dataclass

from prima.allocator.planner import plan, validate_or_reduce


@dataclass(frozen=True)
class BudgetPlan:
    predicted_mb: dict[str, int]
    available_mb: int | None
    raw_budget_mb: dict[str, float]
    final_budget_mib: dict[str, int]
    admitted_roles: list[str]


def calculate_budget_plan(
    predicted_mb: dict[str, int],
    available_mb: int | None,
) -> BudgetPlan:
    raw_budget_mb = plan(predicted_mb, available_mb)
    final_budget_mib, admitted_roles = validate_or_reduce(raw_budget_mb, predicted_mb)
    return BudgetPlan(
        predicted_mb=dict(predicted_mb),
        available_mb=available_mb,
        raw_budget_mb=raw_budget_mb,
        final_budget_mib=final_budget_mib,
        admitted_roles=admitted_roles,
    )


def calculate_budgets(
    predicted_mb: dict[str, int],
    available_mb: int | None,
) -> dict[str, int]:
    return calculate_budget_plan(predicted_mb, available_mb).final_budget_mib

