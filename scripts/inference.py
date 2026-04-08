#!/usr/bin/env python3
"""
inference.py — OpenEnv Supply Chain: LLM Agent Baseline Inference

Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env vars.
Emits strict [START] / [STEP] / [END] structured stdout logs.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="sk-..."
    python inference.py
    python inference.py --task task_1_easy --seed 42
    python inference.py --all-tasks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Ensure local modules are importable ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from env.environment import SupplyChainEnv
from env.models import Action, ActionType
from tasks import TASK_METADATA

# ─────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — using empty API key (will fail against real endpoints).")

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────────────────────────────
# System prompt for the LLM agent
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert supply chain manager operating in a real-world disruption simulation.

You will receive the current state of a multi-echelon supply chain (suppliers, warehouses, retailers, transport legs, active disruptions) and must choose ONE action to take.

Available action types:
- emergency_order       : Instantly order units from a supplier (40% cost premium). Params: supplier_id, destination_node_id (warehouse), units
- reallocate_inventory  : Move stock between warehouses. Params: source_node_id, destination_node_id, units
- expedite_transport    : Halve transit time on a leg (3x cost). Params: transport_leg_id
- activate_backup_route : Enable an inactive transport leg. Params: transport_leg_id OR source_node_id + destination_node_id
- negotiate_supplier    : Attempt to restore a DEGRADED supplier's output (70% success rate). Params: supplier_id
- adjust_safety_stock   : Set safety stock target at a retailer. Params: destination_node_id (retailer), new_safety_stock_days
- reroute_shipment      : Switch a leg to air freight. Params: transport_leg_id
- do_nothing            : Take no action this step.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "action_type": "<one of the above>",
  "supplier_id": "<optional>",
  "destination_node_id": "<optional>",
  "source_node_id": "<optional>",
  "transport_leg_id": "<optional>",
  "units": <optional number>,
  "new_safety_stock_days": <optional number>,
  "notes": "<brief reasoning>"
}

Strategy tips:
- Prioritise activating backup routes when primary routes are blocked
- Negotiate degraded suppliers before spending on emergency orders
- Reallocate inventory from high-stock to low-stock warehouses
- Monitor backlog — stockouts destroy service level score
- In hard tasks, stay under the $500k budget ceiling
"""

def build_user_prompt(obs: dict, task_id: str, step: int) -> str:
    """Build a concise state summary prompt for the LLM."""
    task_info = TASK_METADATA.get(task_id, {})
    suppliers  = obs.get("suppliers", [])
    warehouses = obs.get("warehouses", [])
    retailers  = obs.get("retailers", [])
    legs       = obs.get("transport_legs", [])
    disrupts   = obs.get("active_disruptions", [])
    kpis       = obs.get("kpis", {})

    lines = [
        f"=== SUPPLY CHAIN STATE | Task: {task_id} | Step {step}/{obs.get('max_steps',30)} ===",
        f"Task: {task_info.get('name','?')} ({task_info.get('difficulty','?')})",
        "",
        "── KPIs ──",
        f"  Service Level : {kpis.get('service_level',0)*100:.1f}%",
        f"  Revenue       : ${kpis.get('revenue_usd',0):,.0f}",
        f"  Cost          : ${kpis.get('total_cost_usd',0):,.0f}",
        f"  Profit        : ${kpis.get('profit_usd',0):,.0f}",
        f"  Backlog       : {kpis.get('backlog_units',0):,.0f} units",
        "",
        "── SUPPLIERS ──",
    ]
    for s in suppliers:
        lines.append(
            f"  {s['id']} | {s['name']} | status={s['status']} | "
            f"output={s['current_output_pct']*100:.0f}% | "
            f"cap={s['capacity_units_per_day']:.0f}u/day | cost=${s['unit_cost_usd']}/u"
        )
    lines.append("")
    lines.append("── WAREHOUSES ──")
    for w in warehouses:
        lines.append(
            f"  {w['id']} | {w['name']} | stock={w['current_stock_units']:.0f}u "
            f"/ {w['max_capacity_units']:.0f}u | status={w['status']}"
        )
    lines.append("")
    lines.append("── RETAILERS ──")
    for r in retailers:
        lines.append(
            f"  {r['id']} | {r['name']} | demand={r['daily_demand_units']:.0f}u/day | "
            f"stock={r['current_stock_units']:.0f}u | backlog={r['backlog_units']:.0f}u"
        )
    lines.append("")
    lines.append("── TRANSPORT LEGS ──")
    for l in legs:
        lines.append(
            f"  {l['id']} | {l['origin_id']}→{l['destination_id']} | "
            f"mode={l['mode']} | transit={l['transit_days']}d | "
            f"cost=${l['cost_per_unit_usd']}/u | active={l['active']}"
        )
    lines.append("")
    lines.append("── ACTIVE DISRUPTIONS ──")
    if disrupts:
        for d in disrupts:
            remaining = d['duration_days'] - (step - d['started_at_step'])
            lines.append(
                f"  ⚠  {d['disruption_type']} @ {d['affected_node_id']} | "
                f"sev={d['severity']*100:.0f}% | {remaining}d remaining | {d['description']}"
            )
    else:
        lines.append("  (none)")

    if "budget_ceiling_usd" in obs.get("info", {}):
        lines.append(f"\n⚠  BUDGET CEILING: ${obs['info']['budget_ceiling_usd']:,.0f}")

    lines.append("\nChoose the single best action for this step.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# LLM Action Parsing
# ─────────────────────────────────────────────────────────────────

def parse_llm_action(response_text: str) -> Action:
    """Parse LLM JSON response into a typed Action. Falls back to DO_NOTHING."""
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        data = json.loads(text)
        return Action(**data)
    except Exception as e:
        print(f"[WARN] Failed to parse action: {e} | raw: {response_text[:200]}", file=sys.stderr)
        return Action(action_type=ActionType.DO_NOTHING, notes=f"parse_error: {e}")


def get_llm_action(obs: dict, task_id: str, step: int) -> tuple[Action, str]:
    """Query LLM and return (Action, raw_response_text)."""
    user_prompt = build_user_prompt(obs, task_id, step)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = response.choices[0].message.content or ""
        return parse_llm_action(raw), raw
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
        return Action(action_type=ActionType.DO_NOTHING, notes=f"llm_error: {e}"), f"error: {e}"


# ─────────────────────────────────────────────────────────────────
# Structured Logging — strict [START] / [STEP] / [END] format
# ─────────────────────────────────────────────────────────────────

def log_start(task_id: str, seed: int, model: str, max_steps: int):
    record = {
        "event":    "START",
        "task_id":  task_id,
        "seed":     seed,
        "model":    model,
        "max_steps": max_steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[START] {json.dumps(record)}", flush=True)


def log_step(
    task_id: str,
    step: int,
    action_type: str,
    reward: float,
    done: bool,
    info: dict,
):
    record = {
        "event":       "STEP",
        "task_id":     task_id,
        "step":        step,
        "action_type": action_type,
        "reward":      round(reward, 6),
        "done":        done,
        "service_level": round(info.get("reward_breakdown", {}).get("service_level", 0), 4),
        "cumulative_cost_usd":    round(info.get("cumulative_cost_usd", 0), 2),
        "cumulative_revenue_usd": round(info.get("cumulative_revenue_usd", 0), 2),
        "action_success": info.get("action", {}).get("success", False),
        "action_details": info.get("action", {}).get("details", ""),
    }
    print(f"[STEP] {json.dumps(record)}", flush=True)


def log_end(
    task_id: str,
    seed: int,
    total_steps: int,
    total_reward: float,
    avg_reward: float,
    final_service_level: float,
    final_profit_usd: float,
    backlog_units: float,
    disruptions_resolved: int,
    passed: bool,
    pass_threshold: float,
):
    record = {
        "event":                "END",
        "task_id":              task_id,
        "seed":                 seed,
        "total_steps":          total_steps,
        "total_reward":         round(total_reward, 6),
        "avg_reward":           round(avg_reward, 6),
        "final_service_level":  round(final_service_level, 4),
        "final_profit_usd":     round(final_profit_usd, 2),
        "backlog_units":        round(backlog_units, 1),
        "disruptions_resolved": disruptions_resolved,
        "pass_threshold":       pass_threshold,
        "passed":               passed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[END] {json.dumps(record)}", flush=True)


# ─────────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 42) -> dict:
    """Run one full episode. Returns summary dict."""
    meta       = TASK_METADATA[task_id]
    max_steps  = meta["max_steps"]
    threshold  = meta["pass_threshold"]

    env = SupplyChainEnv(seed=seed)
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.to_dict()

    log_start(task_id=task_id, seed=seed, model=MODEL_NAME, max_steps=max_steps)

    total_reward = 0.0
    step_rewards: List[float] = []
    done         = False
    step         = 0

    while not done:
        # Get action from LLM
        action = Action(action_type=ActionType.DO_NOTHING)
        # Step environment
        result = env.step(action)
        reward = result.reward
        done   = result.done
        info   = result.info

        total_reward += reward
        step_rewards.append(reward)
        step += 1

        log_step(
            task_id=task_id,
            step=step,
            action_type=str(action.action_type),
            reward=reward,
            done=done,
            info=info,
        )

        obs_dict = result.observation

    # Final KPIs from state
    final_state = env.state()
    kpis        = final_state.kpis
    avg_reward  = total_reward / max(len(step_rewards), 1)
    passed      = avg_reward >= threshold

    log_end(
        task_id=task_id,
        seed=seed,
        total_steps=step,
        total_reward=total_reward,
        avg_reward=avg_reward,
        final_service_level=kpis.service_level,
        final_profit_usd=kpis.profit_usd,
        backlog_units=kpis.backlog_units,
        disruptions_resolved=kpis.disruptions_resolved,
        passed=passed,
        pass_threshold=threshold,
    )

    return {
        "task_id":              task_id,
        "seed":                 seed,
        "avg_reward":           avg_reward,
        "final_service_level":  kpis.service_level,
        "final_profit_usd":     kpis.profit_usd,
        "backlog_units":        kpis.backlog_units,
        "disruptions_resolved": kpis.disruptions_resolved,
        "passed":               passed,
        "pass_threshold":       threshold,
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supply Chain OpenEnv — LLM Inference")
    parser.add_argument(
        "--task",
        default="task_1_easy",
        choices=["task_1_easy", "task_2_medium", "task_3_hard"],
        help="Which task to run (default: task_1_easy)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all three tasks sequentially",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tasks = (
        ["task_1_easy", "task_2_medium", "task_3_hard"]
        if args.all_tasks
        else [args.task]
    )

    all_results = []
    for task_id in tasks:
        result = run_episode(task_id=task_id, seed=args.seed)
        all_results.append(result)

    # Summary table to stderr so it doesn't pollute structured stdout
    print("\n" + "=" * 64, file=sys.stderr)
    print("INFERENCE SUMMARY", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"  Model      : {MODEL_NAME}", file=sys.stderr)
    print(f"  Seed       : {args.seed}", file=sys.stderr)
    print(f"  {'Task':<22} {'AvgReward':>10} {'ServiceLv':>10} {'Passed':>8}", file=sys.stderr)
    print("  " + "-" * 52, file=sys.stderr)
    for r in all_results:
        print(
            f"  {r['task_id']:<22} {r['avg_reward']:>10.4f} "
            f"{r['final_service_level']*100:>9.1f}% {str(r['passed']):>8}",
            file=sys.stderr,
        )
    print("=" * 64, file=sys.stderr)


if __name__ == "__main__":
    main()