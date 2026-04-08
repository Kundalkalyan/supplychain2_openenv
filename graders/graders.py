"""
graders.py — Task-specific reward graders for Supply Chain OpenEnv.

Each grader returns reward ∈ [0.0, 1.0] with partial credit throughout
the episode (dense signal, not just binary end-of-episode).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from env.models import Action, ActionType, SupplyChainState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────
# Base Grader
# ─────────────────────────────────────────────

class BaseGrader(ABC):
    """
    OpenEnv agent grader base class.
    score() is called at every step to produce a dense reward signal.
    """

    @abstractmethod
    def score(
        self,
        state: SupplyChainState,
        actions: List[Action],
        cumulative_cost: float,
        cumulative_revenue: float,
    ) -> Tuple[float, Dict[str, Any]]:
        ...

    @staticmethod
    def _service_score(sl: float, target: float) -> float:
        return _clamp(sl / target)

    @staticmethod
    def _profit_score(profit: float, ref: float) -> float:
        if ref <= 0:
            return 0.0
        return _clamp(profit / ref)

    @staticmethod
    def _resolution_score(resolved: int, total: int) -> float:
        return _clamp(resolved / total) if total > 0 else 1.0


# ─────────────────────────────────────────────
# Task 1 Grader — Easy
# ─────────────────────────────────────────────

class Task1Grader(BaseGrader):
    """
    Reward weights:
        0.40  service level ≥ 90%
        0.30  profitability (revenue − cost)
        0.20  disruption resolution
        0.10  action efficiency (avoid excessive emergency orders)

    Partial credit: all sub-scores active from step 1.
    Penalty: consecutive emergency orders reduce efficiency score.
    """
    WEIGHTS = {"service": 0.40, "profit": 0.30, "disruption": 0.20, "efficiency": 0.10}
    MAX_PROFIT_REF = 50_000.0

    def score(self, state, actions, cumulative_cost, cumulative_revenue):
        sl      = state.kpis.service_level if state.kpis else 0.0
        profit  = cumulative_revenue - cumulative_cost
        resolved = len(state.resolved_disruptions)
        total_d  = resolved + len(state.active_disruptions)

        s_service    = self._service_score(sl, 0.90)
        s_profit     = self._profit_score(profit, self.MAX_PROFIT_REF)
        s_disruption = self._resolution_score(resolved, total_d)

        # Efficiency: penalise over-reliance on expensive emergency orders
        emergency_count = sum(1 for a in actions if a.action_type == ActionType.EMERGENCY_ORDER)
        s_efficiency = _clamp(1.0 - emergency_count * 0.04)

        reward = (
            self.WEIGHTS["service"]     * s_service +
            self.WEIGHTS["profit"]      * s_profit +
            self.WEIGHTS["disruption"]  * s_disruption +
            self.WEIGHTS["efficiency"]  * s_efficiency
        )

        info = {
            "service_level": round(sl, 4),
            "service_score": round(s_service, 4),
            "profit_usd": round(profit, 2),
            "profit_score": round(s_profit, 4),
            "disruption_score": round(s_disruption, 4),
            "efficiency_score": round(s_efficiency, 4),
            "total_reward": round(reward, 4),
            "weights": self.WEIGHTS,
        }
        return _clamp(reward), info


# ─────────────────────────────────────────────
# Task 2 Grader — Medium
# ─────────────────────────────────────────────

class Task2Grader(BaseGrader):
    """
    Reward weights:
        0.35  service level ≥ 92% (tighter target due to demand spike)
        0.25  profitability under high air-freight costs
        0.20  warehouse inventory balance (avoid stockout at one hub)
        0.20  disruption resolution speed (early resolution earns bonus)

    Partial credit: balance and speed scores provide signal from step 1.
    Penalty: unbalanced warehouses (>80% or <5% fill) degrade balance score.
    """
    WEIGHTS = {"service": 0.35, "profit": 0.25, "balance": 0.20, "speed": 0.20}
    MAX_PROFIT_REF = 120_000.0

    def score(self, state, actions, cumulative_cost, cumulative_revenue):
        sl     = state.kpis.service_level if state.kpis else 0.0
        profit = cumulative_revenue - cumulative_cost

        # Balance score: each warehouse penalised if fill < 5% or > 85%
        penalties = []
        for wh in state.warehouses:
            fr = wh.fill_ratio
            if fr < 0.05 or fr > 0.85:
                penalties.append(abs(fr - 0.40))
        s_balance = _clamp(1.0 - sum(penalties) / max(len(state.warehouses), 1))

        # Speed bonus: resolved disruptions weighted by how early they're resolved
        steps_elapsed = max(state.step, 1)
        resolved = len(state.resolved_disruptions)
        total_d  = resolved + len(state.active_disruptions)
        speed_raw = (resolved / max(total_d, 1)) * (1.0 - steps_elapsed / state.max_steps)
        s_speed = _clamp(speed_raw * 2.5)

        s_service = self._service_score(sl, 0.92)
        s_profit  = self._profit_score(profit, self.MAX_PROFIT_REF)

        reward = (
            self.WEIGHTS["service"]  * s_service +
            self.WEIGHTS["profit"]   * s_profit +
            self.WEIGHTS["balance"]  * s_balance +
            self.WEIGHTS["speed"]    * s_speed
        )

        info = {
            "service_level": round(sl, 4),
            "service_score": round(s_service, 4),
            "profit_usd": round(profit, 2),
            "profit_score": round(s_profit, 4),
            "balance_score": round(s_balance, 4),
            "speed_score": round(s_speed, 4),
            "total_reward": round(reward, 4),
            "weights": self.WEIGHTS,
        }
        return _clamp(reward), info


# ─────────────────────────────────────────────
# Task 3 Grader — Hard
# ─────────────────────────────────────────────

class Task3Grader(BaseGrader):
    """
    Reward weights:
        0.30  service level ≥ 85% (lower bar due to cascade severity)
        0.25  budget adherence (hard penalty if cost > $500k)
        0.20  triage quality (high-revenue retailers prioritised)
        0.15  supply resilience (active capacity / total capacity)
        0.10  backlog reduction over time

    Hard constraints:
        - Exceeding $500k budget applies exponential penalty
        - Offline supplier chain causes resilience score collapse
    Partial credit: all 5 sub-scores active from step 1.
    """
    WEIGHTS = {"service": 0.30, "budget": 0.25, "triage": 0.20, "resilience": 0.15, "backlog": 0.10}
    BUDGET_CEILING = 500_000.0
    MAX_PROFIT_REF = 200_000.0

    def score(self, state, actions, cumulative_cost, cumulative_revenue):
        sl = state.kpis.service_level if state.kpis else 0.0

        # Budget adherence (linear penalty for overage)
        if cumulative_cost <= self.BUDGET_CEILING:
            s_budget = 1.0
        else:
            over_pct = (cumulative_cost - self.BUDGET_CEILING) / self.BUDGET_CEILING
            s_budget = _clamp(1.0 - over_pct * 2.5)

        # Triage quality: top-revenue retailers should have more stock days
        sorted_r = sorted(state.retailers, key=lambda r: r.revenue_per_unit_usd, reverse=True)
        triage_parts = []
        for rank, r in enumerate(sorted_r):
            target_days = max(7 - rank, 1)
            triage_parts.append(_clamp(r.days_of_stock / target_days))
        s_triage = sum(triage_parts) / max(len(triage_parts), 1)

        # Supply resilience: active capacity / total theoretical capacity
        total_cap  = sum(s.capacity_units_per_day for s in state.suppliers)
        active_cap = sum(s.capacity_units_per_day * s.current_output_pct for s in state.suppliers)
        s_resilience = _clamp(active_cap / max(total_cap, 1))

        # Backlog reduction
        max_possible_backlog = (
            sum(r.daily_demand_units for r in state.retailers) * state.max_steps
        )
        s_backlog = _clamp(1.0 - (state.kpis.backlog_units if state.kpis else 0) / max(max_possible_backlog, 1))

        s_service = self._service_score(sl, 0.85)

        reward = (
            self.WEIGHTS["service"]    * s_service +
            self.WEIGHTS["budget"]     * s_budget +
            self.WEIGHTS["triage"]     * s_triage +
            self.WEIGHTS["resilience"] * s_resilience +
            self.WEIGHTS["backlog"]    * s_backlog
        )

        info = {
            "service_level": round(sl, 4),
            "service_score": round(s_service, 4),
            "budget_used_usd": round(cumulative_cost, 2),
            "budget_ceiling_usd": self.BUDGET_CEILING,
            "budget_score": round(s_budget, 4),
            "triage_score": round(s_triage, 4),
            "resilience_score": round(s_resilience, 4),
            "backlog_score": round(s_backlog, 4),
            "total_reward": round(reward, 4),
            "weights": self.WEIGHTS,
        }
        return _clamp(reward), info


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

GRADER_REGISTRY: Dict[str, BaseGrader] = {
    "task_1_easy":   Task1Grader(),
    "task_2_medium": Task2Grader(),
    "task_3_hard":   Task3Grader(),
}

def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADER_REGISTRY:
        raise ValueError(f"No grader for task '{task_id}'. Available: {list(GRADER_REGISTRY)}")
    return GRADER_REGISTRY[task_id]