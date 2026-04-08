"""
environment.py — SupplyChainEnv: OpenEnv-compliant environment.

Real-world task: Multi-echelon supply chain disruption management.
An AI agent must maintain service levels, control costs, and resolve
disruptions across a global supplier → warehouse → retailer network.

Public API (OpenEnv spec):
    env.reset(task_id, seed) → Observation
    env.step(action)         → StepResult (observation, reward, done, info)
    env.state()              → SupplyChainState
"""
from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, ActionType, DisruptionEvent, DisruptionType,
    KPISnapshot, NodeStatus, Observation, RetailNode,
    StepResult, SupplierNode, SupplyChainState,
    TransportLeg, TransportMode, WarehouseNode,
)
from tasks import TASK_REGISTRY
from graders import get_grader


class SupplyChainEnv:
    """
    OpenEnv-compliant Supply Chain Disruption Management environment.

    Simulates a real-world multi-echelon supply chain under disruption.
    Three tasks ranging from easy (single failure) to hard (cascading
    global failures with a budget ceiling).

    Observation space: structured dict (see Observation model)
    Action space:      typed Action model with 8 discrete action types
    Reward:            dense scalar ∈ [0.0, 1.0] via task-specific grader
    """

    metadata = {
        "name": "supply-chain-disruption",
        "version": "1.0.0",
        "render_modes": ["human", "json"],
    }

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng  = random.Random(seed)
        self._state: Optional[SupplyChainState] = None

    # ──────────────────────────────────────────
    # OpenEnv Public API
    # ──────────────────────────────────────────

    def reset(self, task_id: str = "task_1_easy", seed: Optional[int] = None) -> Observation:
        """Reset environment to initial state. Returns typed Observation."""
        if seed is not None:
            self._seed = seed
            self._rng  = random.Random(seed)

        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY)}")

        builder = TASK_REGISTRY[task_id]
        self._state = builder(rng=self._rng)
        self._state.task_id = task_id
        self._state.cumulative_cost    = 0.0
        self._state.cumulative_revenue = 0.0
        self._state.info["actions_taken"] = []
        return copy.deepcopy(self._state.to_observation())

    def step(self, action: Action) -> StepResult:
        """Advance environment one day. Returns StepResult."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Track action history (for grader efficiency scoring)
        history: list = self._state.info.setdefault("actions_taken", [])
        history.append(action)

        # 1. Apply agent action
        action_cost, action_info = self._apply_action(action)
        self._state.cumulative_cost += action_cost

        # 2. Simulate one day of supply chain flow
        flow_rev, hold_cost, flow_info = self._simulate_day()
        self._state.cumulative_revenue += flow_rev
        self._state.cumulative_cost    += hold_cost

        # 3. Advance disruptions
        self._advance_disruptions()

        # 4. Increment step
        self._state.step += 1
        done = self._state.step >= self._state.max_steps
        self._state.done = done

        # 5. Compute KPIs
        self._state.kpis = self._compute_kpis()

        # 6. Grade (reward)
        grader = get_grader(self._state.task_id)
        reward, reward_info = grader.score(
            state=self._state,
            actions=history,
            cumulative_cost=self._state.cumulative_cost,
            cumulative_revenue=self._state.cumulative_revenue,
        )

        info = {
            "action":             action_info,
            "flow":               flow_info,
            "reward_breakdown":   reward_info,
            "cumulative_cost_usd":    round(self._state.cumulative_cost, 2),
            "cumulative_revenue_usd": round(self._state.cumulative_revenue, 2),
            "step": self._state.step,
        }
        self._state.info.update(info)

        return StepResult(
            observation=self._state.to_observation().to_dict(),
            reward=float(reward),
            done=done,
            truncated=False,
            info=info,
        )

    def state(self) -> SupplyChainState:
        """Return a deep copy of the current full state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    def render(self, mode: str = "human") -> str:
        """Render current state as human-readable text or JSON."""
        if self._state is None:
            return "Environment not initialised. Call reset() first."
        s = self._state
        k = s.kpis
        if mode == "json":
            import json
            return json.dumps(s.to_observation().to_dict(), indent=2, default=str)
        lines = [
            f"╔═══ Supply Chain │ {s.task_id} │ Step {s.step}/{s.max_steps} ═══╗",
            f"  Service Level : {k.service_level*100:6.1f}%",
            f"  Revenue       : ${k.revenue_usd:>12,.0f}",
            f"  Cost          : ${k.total_cost_usd:>12,.0f}",
            f"  Profit        : ${k.profit_usd:>12,.0f}",
            f"  Backlog       : {k.backlog_units:>12,.0f} units",
            f"  Active disrupt: {k.disruptions_active}",
        ]
        for d in s.active_disruptions:
            remaining = d.duration_days - (s.step - d.started_at_step)
            lines.append(f"  ⚠  {d.disruption_type} @ {d.affected_node_id} "
                         f"(sev={d.severity:.0%}, {remaining}d left)")
        lines.append("╚" + "═" * 48 + "╝")
        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Action Application
    # ──────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[float, dict]:
        s = self._state
        cost = 0.0
        info: Dict[str, Any] = {"type": action.action_type, "success": False, "details": ""}

        if action.action_type == ActionType.DO_NOTHING:
            info.update(success=True, details="No action taken.")
            return 0.0, info

        elif action.action_type == ActionType.EMERGENCY_ORDER:
            sup = self._supplier(action.supplier_id)
            wh  = self._warehouse(action.destination_node_id)
            if sup and wh and sup.status != NodeStatus.OFFLINE:
                units     = min(action.units or 100, sup.capacity_units_per_day * 2)
                cost      = units * sup.unit_cost_usd * 1.40  # 40% premium
                wh.current_stock_units = min(
                    wh.current_stock_units + units, wh.max_capacity_units
                )
                info.update(success=True, details=f"Emergency order {units:.0f}u → {wh.name}. Cost ${cost:,.0f}")
            else:
                info["details"] = "Failed: supplier offline or not found."

        elif action.action_type == ActionType.REALLOCATE_INVENTORY:
            src = self._warehouse(action.source_node_id)
            dst = self._warehouse(action.destination_node_id)
            if src and dst and action.units:
                transfer = min(
                    action.units,
                    src.current_stock_units,
                    dst.max_capacity_units - dst.current_stock_units,
                )
                src.current_stock_units -= transfer
                dst.current_stock_units += transfer
                cost = transfer * 0.50
                info.update(success=True, details=f"Moved {transfer:.0f}u: {src.name}→{dst.name}. Cost ${cost:,.0f}")
            else:
                info["details"] = "Failed: nodes not found or insufficient stock."

        elif action.action_type == ActionType.EXPEDITE_TRANSPORT:
            leg = self._leg(action.transport_leg_id)
            if leg:
                leg.transit_days = max(1, leg.transit_days // 2)
                leg.cost_per_unit_usd *= 3.0
                cost = 500.0
                info.update(success=True, details=f"Expedited {leg.id}: transit→{leg.transit_days}d. Fee $500")
            else:
                info["details"] = "Failed: leg not found."

        elif action.action_type == ActionType.ACTIVATE_BACKUP_ROUTE:
            for leg in s.transport_legs:
                if not leg.active and (
                    leg.origin_id      == action.source_node_id or
                    leg.destination_id == action.destination_node_id or
                    leg.id             == action.transport_leg_id
                ):
                    leg.active = True
                    cost = 200.0
                    info.update(success=True, details=f"Activated backup route {leg.id}. Fee $200")
                    break
            else:
                info["details"] = "Failed: no matching inactive backup route found."

        elif action.action_type == ActionType.NEGOTIATE_SUPPLIER:
            sup = self._supplier(action.supplier_id)
            if sup and sup.status == NodeStatus.DEGRADED:
                if self._rng.random() < 0.70:
                    sup.current_output_pct = min(1.0, sup.current_output_pct + 0.50)
                    if sup.current_output_pct >= 0.5:
                        sup.status = NodeStatus.OPERATIONAL
                    cost = 1_000.0
                    info.update(success=True, details=f"Negotiation OK: {sup.name} → {sup.current_output_pct:.0%} output.")
                else:
                    cost = 500.0
                    info.update(success=False, details=f"Negotiation failed: {sup.name} unresponsive.")
            else:
                info["details"] = "Failed: supplier not degraded or not found."

        elif action.action_type == ActionType.ADJUST_SAFETY_STOCK:
            ret = self._retailer(action.destination_node_id)
            if ret and action.new_safety_stock_days is not None:
                old = ret.safety_stock_days
                ret.safety_stock_days = max(0, action.new_safety_stock_days)
                info.update(success=True, details=f"{ret.name}: safety stock {old:.1f}d→{ret.safety_stock_days:.1f}d")
            else:
                info["details"] = "Failed: retailer not found."

        elif action.action_type == ActionType.REROUTE_SHIPMENT:
            leg = self._leg(action.transport_leg_id)
            if leg:
                leg.mode = TransportMode.AIR
                leg.transit_days = max(1, leg.transit_days - 3)
                leg.cost_per_unit_usd *= 2.5
                cost = 300.0
                info.update(success=True, details=f"Rerouted {leg.id} to AIR. Transit→{leg.transit_days}d. Fee $300")
            else:
                info["details"] = "Failed: leg not found."

        return cost, info

    # ──────────────────────────────────────────
    # Day Simulation
    # ──────────────────────────────────────────

    def _simulate_day(self) -> Tuple[float, float, dict]:
        s = self._state
        revenue     = 0.0
        holding_cost = 0.0

        # Supplier → Warehouse replenishment
        for wh in s.warehouses:
            if wh.status == NodeStatus.OFFLINE:
                continue
            throughput_mult = 0.5 if wh.status == NodeStatus.DEGRADED else 1.0
            for leg in s.transport_legs:
                if not leg.active or leg.destination_id != wh.id:
                    continue
                sup = self._supplier(leg.origin_id)
                if not sup or sup.status == NodeStatus.OFFLINE:
                    continue
                daily = (
                    sup.capacity_units_per_day
                    * sup.current_output_pct
                    * throughput_mult
                    / max(leg.transit_days, 1)
                )
                if self._rng.random() < leg.reliability:
                    received = min(daily, wh.max_capacity_units - wh.current_stock_units)
                    received = max(0.0, received)
                    wh.current_stock_units += received
                    self._state.cumulative_cost += received * leg.cost_per_unit_usd

            # Warehouse → Warehouse (inter-hub)
            # also handles legs where origin is another warehouse
        for wh in s.warehouses:
            for leg in s.transport_legs:
                if not leg.active or leg.destination_id != wh.id:
                    continue
                src_wh = self._warehouse(leg.origin_id)
                if not src_wh:
                    continue
                transfer = min(
                    leg.max_units_per_shipment / max(leg.transit_days, 1),
                    src_wh.current_stock_units * 0.10,  # ship up to 10% per day
                    wh.max_capacity_units - wh.current_stock_units,
                )
                transfer = max(0.0, transfer)
                if self._rng.random() < leg.reliability:
                    src_wh.current_stock_units -= transfer
                    wh.current_stock_units     += transfer
                    self._state.cumulative_cost += transfer * leg.cost_per_unit_usd

        # Warehouse → Retailer fulfilment
        all_warehouses = list(s.warehouses)
        for ret in s.retailers:
            variance     = self._rng.gauss(0, ret.daily_demand_units * ret.demand_variance_pct)
            actual_demand = max(0.0, ret.daily_demand_units + variance)
            fulfilled = 0.0
            for wh in all_warehouses:
                if wh.status == NodeStatus.OFFLINE:
                    continue
                needed  = actual_demand - fulfilled
                can_give = min(needed, wh.current_stock_units)
                if can_give > 0 and (
                    ret.id in wh.outbound_nodes or not wh.outbound_nodes
                ):
                    wh.current_stock_units -= can_give
                    fulfilled += can_give
                if fulfilled >= actual_demand:
                    break
            revenue += fulfilled * ret.revenue_per_unit_usd
            ret.backlog_units += max(0.0, actual_demand - fulfilled)
            ret.current_stock_units = max(0.0, ret.current_stock_units - fulfilled)

        # Holding costs
        for wh in s.warehouses:
            holding_cost += wh.current_stock_units * wh.holding_cost_per_unit_per_day

        return revenue, holding_cost, {
            "day_revenue_usd": round(revenue, 2),
            "holding_cost_usd": round(holding_cost, 2),
        }

    # ──────────────────────────────────────────
    # Disruption Lifecycle
    # ──────────────────────────────────────────

    def _advance_disruptions(self):
        s = self._state
        still_active = []
        for d in s.active_disruptions:
            age = s.step - d.started_at_step
            if age >= d.duration_days:
                d.resolved = True
                self._restore_node(d.affected_node_id)
                s.resolved_disruptions.append(d)
            else:
                still_active.append(d)
        s.active_disruptions = still_active

        # Small random disruption chance after step 3
        if s.step > 3 and self._rng.random() < 0.04:
            self._spawn_random_disruption()

    def _spawn_random_disruption(self):
        s = self._state
        candidates = [sup.id for sup in s.suppliers if sup.status == NodeStatus.OPERATIONAL]
        if not candidates:
            return
        node_id = self._rng.choice(candidates)
        dtype   = self._rng.choice([DisruptionType.TRANSPORT_DELAY, DisruptionType.WEATHER_EVENT])
        sev     = self._rng.uniform(0.10, 0.40)
        event   = DisruptionEvent(
            disruption_type=dtype,
            affected_node_id=node_id,
            severity=sev,
            duration_days=self._rng.randint(1, 4),
            description=f"Random {dtype} at {node_id} (sev={sev:.0%}).",
            started_at_step=s.step,
        )
        # Apply
        for sup in s.suppliers:
            if sup.id == node_id:
                sup.current_output_pct = max(0.0, sup.current_output_pct - sev)
                sup.status = NodeStatus.DEGRADED if sup.current_output_pct > 0.1 else NodeStatus.OFFLINE
        s.active_disruptions.append(event)

    def _restore_node(self, node_id: str):
        for sup in self._state.suppliers:
            if sup.id == node_id:
                sup.status = NodeStatus.OPERATIONAL
                sup.current_output_pct = 1.0
                return
        for wh in self._state.warehouses:
            if wh.id == node_id:
                wh.status = NodeStatus.OPERATIONAL
                return

    # ──────────────────────────────────────────
    # KPI Computation
    # ──────────────────────────────────────────

    def _compute_kpis(self) -> KPISnapshot:
        s = self._state
        total_demand  = sum(r.daily_demand_units for r in s.retailers) * max(s.step, 1)
        total_backlog = sum(r.backlog_units for r in s.retailers)
        service_level = max(0.0, 1.0 - total_backlog / max(total_demand, 1))

        total_stock = sum(w.current_stock_units for w in s.warehouses)
        daily_demand = sum(r.daily_demand_units for r in s.retailers)
        avg_days = total_stock / max(daily_demand, 1)

        return KPISnapshot(
            service_level=min(service_level, 1.0),
            total_cost_usd=s.cumulative_cost,
            avg_days_of_stock=avg_days,
            disruptions_active=len(s.active_disruptions),
            disruptions_resolved=len(s.resolved_disruptions),
            backlog_units=total_backlog,
            revenue_usd=s.cumulative_revenue,
            profit_usd=s.cumulative_revenue - s.cumulative_cost,
        )

    # ──────────────────────────────────────────
    # Lookup Helpers
    # ──────────────────────────────────────────

    def _supplier(self, sid) -> Optional[SupplierNode]:
        return next((x for x in self._state.suppliers if x.id == sid), None) if sid else None

    def _warehouse(self, wid) -> Optional[WarehouseNode]:
        return next((x for x in self._state.warehouses if x.id == wid), None) if wid else None

    def _retailer(self, rid) -> Optional[RetailNode]:
        return next((x for x in self._state.retailers if x.id == rid), None) if rid else None

    def _leg(self, lid) -> Optional[TransportLeg]:
        return next((x for x in self._state.transport_legs if x.id == lid), None) if lid else None