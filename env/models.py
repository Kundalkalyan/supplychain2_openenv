"""
models.py — Typed Pydantic models for Supply Chain OpenEnv.
Full OpenEnv spec compliance: Observation, Action, Reward models.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class NodeStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED    = "degraded"
    OFFLINE     = "offline"

class TransportMode(str, Enum):
    SEA   = "sea"
    AIR   = "air"
    RAIL  = "rail"
    TRUCK = "truck"

class DisruptionType(str, Enum):
    PORT_CLOSURE     = "port_closure"
    SUPPLIER_FAILURE = "supplier_failure"
    TRANSPORT_DELAY  = "transport_delay"
    DEMAND_SPIKE     = "demand_spike"
    WEATHER_EVENT    = "weather_event"
    QUALITY_RECALL   = "quality_recall"

class ActionType(str, Enum):
    EMERGENCY_ORDER       = "emergency_order"
    REALLOCATE_INVENTORY  = "reallocate_inventory"
    EXPEDITE_TRANSPORT    = "expedite_transport"
    ACTIVATE_BACKUP_ROUTE = "activate_backup_route"
    NEGOTIATE_SUPPLIER    = "negotiate_supplier"
    ADJUST_SAFETY_STOCK   = "adjust_safety_stock"
    REROUTE_SHIPMENT      = "reroute_shipment"
    DO_NOTHING            = "do_nothing"


# ─────────────────────────────────────────────
# Network Nodes
# ─────────────────────────────────────────────

class SupplierNode(BaseModel):
    id: str
    name: str
    country: str
    capacity_units_per_day: float
    reliability_score: float = Field(ge=0.0, le=1.0)
    lead_time_days: int
    status: NodeStatus = NodeStatus.OPERATIONAL
    current_output_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    unit_cost_usd: float
    specialisations: List[str] = Field(default_factory=list)

class WarehouseNode(BaseModel):
    id: str
    name: str
    location: str
    max_capacity_units: float
    current_stock_units: float
    holding_cost_per_unit_per_day: float
    inbound_nodes: List[str] = Field(default_factory=list)
    outbound_nodes: List[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.OPERATIONAL

    @property
    def fill_ratio(self) -> float:
        return self.current_stock_units / max(self.max_capacity_units, 1)

class RetailNode(BaseModel):
    id: str
    name: str
    location: str
    daily_demand_units: float
    demand_variance_pct: float = 0.10
    safety_stock_days: float = 7.0
    current_stock_units: float = 0.0
    service_level_target: float = 0.95
    backlog_units: float = 0.0
    revenue_per_unit_usd: float = 0.0

    @property
    def days_of_stock(self) -> float:
        return self.current_stock_units / max(self.daily_demand_units, 1)

class TransportLeg(BaseModel):
    id: str
    origin_id: str
    destination_id: str
    mode: TransportMode
    transit_days: int
    cost_per_unit_usd: float
    max_units_per_shipment: float
    reliability: float = Field(ge=0.0, le=1.0, default=0.95)
    active: bool = True


# ─────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────

class DisruptionEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    disruption_type: DisruptionType
    affected_node_id: str
    severity: float = Field(ge=0.0, le=1.0)
    duration_days: int
    description: str
    financial_impact_usd: float = 0.0
    started_at_step: int = 0
    resolved: bool = False


# ─────────────────────────────────────────────
# OpenEnv Core Models
# ─────────────────────────────────────────────

class KPISnapshot(BaseModel):
    service_level: float
    total_cost_usd: float
    avg_days_of_stock: float
    disruptions_active: int
    disruptions_resolved: int
    backlog_units: float
    revenue_usd: float
    profit_usd: float

class Observation(BaseModel):
    """OpenEnv typed Observation model."""
    step: int = 0
    max_steps: int = 30
    task_id: str = "task_1_easy"
    suppliers: List[SupplierNode] = Field(default_factory=list)
    warehouses: List[WarehouseNode] = Field(default_factory=list)
    retailers: List[RetailNode] = Field(default_factory=list)
    transport_legs: List[TransportLeg] = Field(default_factory=list)
    active_disruptions: List[DisruptionEvent] = Field(default_factory=list)
    kpis: Optional[KPISnapshot] = None
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class Action(BaseModel):
    """OpenEnv typed Action model."""
    action_type: ActionType
    source_node_id: Optional[str] = None
    destination_node_id: Optional[str] = None
    transport_leg_id: Optional[str] = None
    units: Optional[float] = None
    supplier_id: Optional[str] = None
    new_safety_stock_days: Optional[float] = None
    notes: Optional[str] = None

class Reward(BaseModel):
    """OpenEnv typed Reward model."""
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    done: bool = False
    truncated: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

class StepResult(BaseModel):
    """Complete return type from env.step()."""
    observation: Dict[str, Any]
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class SupplyChainState(BaseModel):
    """Full internal state (returned by env.state())."""
    step: int = 0
    max_steps: int = 30
    task_id: str = "task_1_easy"
    suppliers: List[SupplierNode] = Field(default_factory=list)
    warehouses: List[WarehouseNode] = Field(default_factory=list)
    retailers: List[RetailNode] = Field(default_factory=list)
    transport_legs: List[TransportLeg] = Field(default_factory=list)
    active_disruptions: List[DisruptionEvent] = Field(default_factory=list)
    resolved_disruptions: List[DisruptionEvent] = Field(default_factory=list)
    kpis: Optional[KPISnapshot] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
    cumulative_cost: float = 0.0
    cumulative_revenue: float = 0.0

    def to_observation(self) -> Observation:
        return Observation(
            step=self.step,
            max_steps=self.max_steps,
            task_id=self.task_id,
            suppliers=self.suppliers,
            warehouses=self.warehouses,
            retailers=self.retailers,
            transport_legs=self.transport_legs,
            active_disruptions=self.active_disruptions,
            kpis=self.kpis,
            done=self.done,
        )