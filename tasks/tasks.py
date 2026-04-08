"""
tasks.py — Task definitions and registry for Supply Chain OpenEnv.

task_1_easy   — Single supplier failure, 20 steps, small network
task_2_medium — Port strike + demand spike, 25 steps, mid-size network  
task_3_hard   — Cascading global failures + budget ceiling, 30 steps
"""
from __future__ import annotations
import random
from typing import Callable, Dict
from env.models import (
    DisruptionEvent, DisruptionType, KPISnapshot, NodeStatus,
    RetailNode, SupplierNode, SupplyChainState,
    TransportLeg, TransportMode, WarehouseNode,
)


# ─────────────────────────────────────────────
# Task 1 — EASY: Single Supplier Failure
# ─────────────────────────────────────────────

def build_task_1(rng: random.Random) -> SupplyChainState:
    """
    Scenario: Primary supplier Beta Parts Co has a factory fire.
    Output drops 70%. Agent must restore service above 90% in 20 days.
    Network: 2 suppliers, 1 warehouse, 2 retailers.
    """
    suppliers = [
        SupplierNode(
            id="SUP-A", name="Alpha Components", country="Germany",
            capacity_units_per_day=500, reliability_score=0.95,
            lead_time_days=3, unit_cost_usd=10.0,
            specialisations=["electronics"],
        ),
        SupplierNode(
            id="SUP-B", name="Beta Parts Co", country="Poland",
            capacity_units_per_day=300, reliability_score=0.88,
            lead_time_days=4, unit_cost_usd=9.0,
            specialisations=["electronics"],
            status=NodeStatus.DEGRADED,
            current_output_pct=0.30,
        ),
    ]
    warehouses = [
        WarehouseNode(
            id="WH-EU", name="Central EU Hub", location="Frankfurt, DE",
            max_capacity_units=5000, current_stock_units=1200,
            holding_cost_per_unit_per_day=0.05,
            inbound_nodes=["SUP-A", "SUP-B"],
            outbound_nodes=["RET-1", "RET-2"],
        ),
    ]
    retailers = [
        RetailNode(
            id="RET-1", name="Berlin Electronics", location="Berlin, DE",
            daily_demand_units=120, current_stock_units=300,
            revenue_per_unit_usd=25.0,
        ),
        RetailNode(
            id="RET-2", name="Munich Tech Store", location="Munich, DE",
            daily_demand_units=80, current_stock_units=200,
            revenue_per_unit_usd=25.0,
        ),
    ]
    legs = [
        TransportLeg(
            id="LEG-A-EU", origin_id="SUP-A", destination_id="WH-EU",
            mode=TransportMode.TRUCK, transit_days=2,
            cost_per_unit_usd=0.80, max_units_per_shipment=500,
            active=True,
        ),
        TransportLeg(
            id="LEG-B-EU", origin_id="SUP-B", destination_id="WH-EU",
            mode=TransportMode.TRUCK, transit_days=3,
            cost_per_unit_usd=0.70, max_units_per_shipment=300,
            active=False,  # backup — currently inactive
        ),
    ]
    disruptions = [
        DisruptionEvent(
            disruption_type=DisruptionType.SUPPLIER_FAILURE,
            affected_node_id="SUP-B",
            severity=0.70,
            duration_days=8,
            description="Beta Parts Co factory fire — output reduced to 30%.",
            financial_impact_usd=50_000,
            started_at_step=0,
        ),
    ]
    kpis = KPISnapshot(
        service_level=1.0, total_cost_usd=0, avg_days_of_stock=10,
        disruptions_active=1, disruptions_resolved=0,
        backlog_units=0, revenue_usd=0, profit_usd=0,
    )
    return SupplyChainState(
        max_steps=20, task_id="task_1_easy",
        suppliers=suppliers, warehouses=warehouses,
        retailers=retailers, transport_legs=legs,
        active_disruptions=disruptions, kpis=kpis,
    )


# ─────────────────────────────────────────────
# Task 2 — MEDIUM: Port Strike + Demand Spike
# ─────────────────────────────────────────────

def build_task_2(rng: random.Random) -> SupplyChainState:
    """
    Scenario: LA port strike halts all sea freight AND NYC demand spikes 50%.
    Agent must use air freight, balance 2 warehouses, stay profitable.
    Network: 3 suppliers, 2 warehouses, 3 retailers.
    """
    suppliers = [
        SupplierNode(
            id="SUP-CN", name="Shenzhen Manufacturing", country="China",
            capacity_units_per_day=1000, reliability_score=0.90,
            lead_time_days=2, unit_cost_usd=6.0,
            specialisations=["consumer_goods"],
        ),
        SupplierNode(
            id="SUP-IN", name="Mumbai Textiles", country="India",
            capacity_units_per_day=600, reliability_score=0.85,
            lead_time_days=3, unit_cost_usd=5.5,
            specialisations=["consumer_goods"],
        ),
        SupplierNode(
            id="SUP-US", name="Ohio Assembly", country="USA",
            capacity_units_per_day=400, reliability_score=0.92,
            lead_time_days=1, unit_cost_usd=14.0,
            specialisations=["consumer_goods"],
        ),
    ]
    warehouses = [
        WarehouseNode(
            id="WH-LA", name="LA Distribution", location="Los Angeles, USA",
            max_capacity_units=8000, current_stock_units=2000,
            holding_cost_per_unit_per_day=0.08,
            inbound_nodes=["SUP-CN", "SUP-IN"],
            outbound_nodes=["RET-NYC", "RET-CHI"],
            status=NodeStatus.DEGRADED,
        ),
        WarehouseNode(
            id="WH-NY", name="Newark Warehouse", location="Newark, USA",
            max_capacity_units=5000, current_stock_units=800,
            holding_cost_per_unit_per_day=0.10,
            inbound_nodes=["SUP-US"],
            outbound_nodes=["RET-NYC", "RET-BOS"],
        ),
    ]
    retailers = [
        RetailNode(
            id="RET-NYC", name="New York Flagship", location="New York, USA",
            daily_demand_units=450,  # spiked from 300
            demand_variance_pct=0.20,
            current_stock_units=600, revenue_per_unit_usd=35.0,
        ),
        RetailNode(
            id="RET-CHI", name="Chicago Store", location="Chicago, USA",
            daily_demand_units=200, current_stock_units=400,
            revenue_per_unit_usd=30.0,
        ),
        RetailNode(
            id="RET-BOS", name="Boston Outlet", location="Boston, USA",
            daily_demand_units=150, current_stock_units=250,
            revenue_per_unit_usd=28.0,
        ),
    ]
    legs = [
        TransportLeg(
            id="LEG-CN-LA-SEA", origin_id="SUP-CN", destination_id="WH-LA",
            mode=TransportMode.SEA, transit_days=14,
            cost_per_unit_usd=1.20, max_units_per_shipment=2000,
            active=False,  # port closed
        ),
        TransportLeg(
            id="LEG-CN-LA-AIR", origin_id="SUP-CN", destination_id="WH-LA",
            mode=TransportMode.AIR, transit_days=2,
            cost_per_unit_usd=6.00, max_units_per_shipment=500,
            active=False,  # must be activated
        ),
        TransportLeg(
            id="LEG-IN-LA-SEA", origin_id="SUP-IN", destination_id="WH-LA",
            mode=TransportMode.SEA, transit_days=18,
            cost_per_unit_usd=1.10, max_units_per_shipment=1500,
            active=False,  # port closed
        ),
        TransportLeg(
            id="LEG-US-NY", origin_id="SUP-US", destination_id="WH-NY",
            mode=TransportMode.TRUCK, transit_days=1,
            cost_per_unit_usd=2.00, max_units_per_shipment=400,
            active=True,
        ),
        TransportLeg(
            id="LEG-LA-NY", origin_id="WH-LA", destination_id="WH-NY",
            mode=TransportMode.TRUCK, transit_days=3,
            cost_per_unit_usd=1.50, max_units_per_shipment=1000,
            active=True,
        ),
    ]
    disruptions = [
        DisruptionEvent(
            disruption_type=DisruptionType.PORT_CLOSURE,
            affected_node_id="WH-LA",
            severity=0.80,
            duration_days=10,
            description="LA port strike — all sea freight halted for 10 days.",
            financial_impact_usd=250_000,
            started_at_step=0,
        ),
        DisruptionEvent(
            disruption_type=DisruptionType.DEMAND_SPIKE,
            affected_node_id="RET-NYC",
            severity=0.50,
            duration_days=6,
            description="Viral campaign drives +50% demand at NYC flagship.",
            financial_impact_usd=0,
            started_at_step=0,
        ),
    ]
    kpis = KPISnapshot(
        service_level=1.0, total_cost_usd=0, avg_days_of_stock=8,
        disruptions_active=2, disruptions_resolved=0,
        backlog_units=0, revenue_usd=0, profit_usd=0,
    )
    return SupplyChainState(
        max_steps=25, task_id="task_2_medium",
        suppliers=suppliers, warehouses=warehouses,
        retailers=retailers, transport_legs=legs,
        active_disruptions=disruptions, kpis=kpis,
    )


# ─────────────────────────────────────────────
# Task 3 — HARD: Cascading Global Failures
# ─────────────────────────────────────────────

def build_task_3(rng: random.Random) -> SupplyChainState:
    """
    Scenario: Taiwan offline (20d), Korea quality recall (7d), Singapore typhoon (5d).
    Hard $500k budget ceiling. Triage scarce supply across global retailers.
    Network: 4 suppliers, 3 warehouses, 4 retailers.
    """
    suppliers = [
        SupplierNode(
            id="SUP-JP", name="Tokyo Precision", country="Japan",
            capacity_units_per_day=800, reliability_score=0.97,
            lead_time_days=5, unit_cost_usd=18.0,
            specialisations=["high_precision"],
        ),
        SupplierNode(
            id="SUP-KR", name="Seoul Semiconductors", country="South Korea",
            capacity_units_per_day=600, reliability_score=0.93,
            lead_time_days=4, unit_cost_usd=20.0,
            specialisations=["semiconductors"],
            status=NodeStatus.DEGRADED,
            current_output_pct=0.20,
        ),
        SupplierNode(
            id="SUP-TW", name="Taipei Chips", country="Taiwan",
            capacity_units_per_day=1200, reliability_score=0.91,
            lead_time_days=3, unit_cost_usd=15.0,
            specialisations=["semiconductors", "high_precision"],
            status=NodeStatus.OFFLINE,
            current_output_pct=0.0,
        ),
        SupplierNode(
            id="SUP-EU", name="Dresden Fab", country="Germany",
            capacity_units_per_day=400, reliability_score=0.94,
            lead_time_days=2, unit_cost_usd=28.0,
            specialisations=["semiconductors"],
        ),
    ]
    warehouses = [
        WarehouseNode(
            id="WH-SG", name="Singapore Hub", location="Singapore",
            max_capacity_units=10000, current_stock_units=1500,
            holding_cost_per_unit_per_day=0.12,
            inbound_nodes=["SUP-JP", "SUP-KR", "SUP-TW"],
            outbound_nodes=["WH-EU2", "WH-US"],
            status=NodeStatus.DEGRADED,
        ),
        WarehouseNode(
            id="WH-EU2", name="Rotterdam Hub", location="Netherlands",
            max_capacity_units=6000, current_stock_units=500,
            holding_cost_per_unit_per_day=0.14,
            inbound_nodes=["SUP-EU", "WH-SG"],
            outbound_nodes=["RET-EU1", "RET-EU2"],
        ),
        WarehouseNode(
            id="WH-US", name="Dallas Hub", location="Texas, USA",
            max_capacity_units=7000, current_stock_units=800,
            holding_cost_per_unit_per_day=0.10,
            inbound_nodes=["WH-SG"],
            outbound_nodes=["RET-US1", "RET-US2"],
        ),
    ]
    retailers = [
        RetailNode(
            id="RET-EU1", name="Frankfurt B2B", location="Germany",
            daily_demand_units=180, current_stock_units=200,
            revenue_per_unit_usd=60.0, safety_stock_days=5,
        ),
        RetailNode(
            id="RET-EU2", name="Paris Consumer", location="France",
            daily_demand_units=120, current_stock_units=100,
            revenue_per_unit_usd=55.0, safety_stock_days=5,
        ),
        RetailNode(
            id="RET-US1", name="Austin Tech", location="Texas, USA",
            daily_demand_units=200, current_stock_units=300,
            revenue_per_unit_usd=65.0, safety_stock_days=7,
        ),
        RetailNode(
            id="RET-US2", name="Seattle Enterprise", location="Washington, USA",
            daily_demand_units=150, current_stock_units=180,
            revenue_per_unit_usd=70.0, safety_stock_days=7,
        ),
    ]
    legs = [
        TransportLeg(
            id="LEG-JP-SG", origin_id="SUP-JP", destination_id="WH-SG",
            mode=TransportMode.SEA, transit_days=4,
            cost_per_unit_usd=2.50, max_units_per_shipment=800, reliability=0.90, active=True,
        ),
        TransportLeg(
            id="LEG-KR-SG", origin_id="SUP-KR", destination_id="WH-SG",
            mode=TransportMode.SEA, transit_days=3,
            cost_per_unit_usd=2.20, max_units_per_shipment=600, reliability=0.85, active=True,
        ),
        TransportLeg(
            id="LEG-TW-SG", origin_id="SUP-TW", destination_id="WH-SG",
            mode=TransportMode.SEA, transit_days=2,
            cost_per_unit_usd=1.80, max_units_per_shipment=1200, active=False,
        ),
        TransportLeg(
            id="LEG-EU-WHEU", origin_id="SUP-EU", destination_id="WH-EU2",
            mode=TransportMode.TRUCK, transit_days=1,
            cost_per_unit_usd=3.00, max_units_per_shipment=400, active=True,
        ),
        TransportLeg(
            id="LEG-SG-WHEU", origin_id="WH-SG", destination_id="WH-EU2",
            mode=TransportMode.SEA, transit_days=18,
            cost_per_unit_usd=3.50, max_units_per_shipment=2000, active=True,
        ),
        TransportLeg(
            id="LEG-SG-WHUS", origin_id="WH-SG", destination_id="WH-US",
            mode=TransportMode.SEA, transit_days=14,
            cost_per_unit_usd=3.00, max_units_per_shipment=2500, active=True,
        ),
        TransportLeg(
            id="LEG-SG-WHEU-AIR", origin_id="WH-SG", destination_id="WH-EU2",
            mode=TransportMode.AIR, transit_days=2,
            cost_per_unit_usd=12.0, max_units_per_shipment=400, active=False,
        ),
    ]
    disruptions = [
        DisruptionEvent(
            disruption_type=DisruptionType.SUPPLIER_FAILURE,
            affected_node_id="SUP-TW",
            severity=1.0,
            duration_days=20,
            description="Taipei Chips: geopolitical shutdown — total halt for 20 days.",
            financial_impact_usd=800_000,
            started_at_step=0,
        ),
        DisruptionEvent(
            disruption_type=DisruptionType.QUALITY_RECALL,
            affected_node_id="SUP-KR",
            severity=0.80,
            duration_days=7,
            description="Seoul Semiconductors: defective batch recall — output capped at 20%.",
            financial_impact_usd=200_000,
            started_at_step=0,
        ),
        DisruptionEvent(
            disruption_type=DisruptionType.WEATHER_EVENT,
            affected_node_id="WH-SG",
            severity=0.40,
            duration_days=5,
            description="Typhoon limits Singapore hub throughput by 40%.",
            financial_impact_usd=120_000,
            started_at_step=0,
        ),
    ]
    kpis = KPISnapshot(
        service_level=1.0, total_cost_usd=0, avg_days_of_stock=4,
        disruptions_active=3, disruptions_resolved=0,
        backlog_units=0, revenue_usd=0, profit_usd=0,
    )
    state = SupplyChainState(
        max_steps=30, task_id="task_3_hard",
        suppliers=suppliers, warehouses=warehouses,
        retailers=retailers, transport_legs=legs,
        active_disruptions=disruptions, kpis=kpis,
    )
    state.info["budget_ceiling_usd"] = 500_000
    return state


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Callable] = {
    "task_1_easy":   build_task_1,
    "task_2_medium": build_task_2,
    "task_3_hard":   build_task_3,
}

TASK_METADATA: Dict[str, dict] = {
    "task_1_easy": {
        "name": "Single Supplier Failure",
        "difficulty": "easy",
        "max_steps": 20,
        "pass_threshold": 0.70,
        "description": (
            "One supplier has a factory fire, output drops 70%. "
            "Agent must maintain service level ≥ 90% via negotiation, "
            "emergency orders, or backup routes."
        ),
    },
    "task_2_medium": {
        "name": "Port Strike & Demand Spike",
        "difficulty": "medium",
        "max_steps": 25,
        "pass_threshold": 0.60,
        "description": (
            "LA port strike halts sea freight while NYC demand spikes +50%. "
            "Agent must activate air freight, balance two warehouses, "
            "and remain profitable."
        ),
    },
    "task_3_hard": {
        "name": "Cascading Global Failures",
        "difficulty": "hard",
        "max_steps": 30,
        "pass_threshold": 0.45,
        "description": (
            "Taiwan offline 20 days, Korea quality recall, Singapore typhoon — all concurrent. "
            "Hard $500k budget ceiling. Agent triages scarce semiconductors across 4 global retailers."
        ),
    },
}