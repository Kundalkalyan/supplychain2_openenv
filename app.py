"""
app.py — FastAPI server implementing the OpenEnv HTTP API.

Endpoints:
    GET  /              → Landing page
    GET  /health        → Health check
    GET  /tasks         → List all tasks with metadata
    POST /reset         → Reset episode, returns Observation
    POST /step          → Step with action, returns StepResult
    GET  /state         → Current full state
    GET  /openenv.yaml  → OpenEnv spec file
    GET  /docs          → Swagger UI (auto-generated)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from environment import SupplyChainEnv
from env.models import Action, ActionType
from tasks import TASK_METADATA


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain Disruption — OpenEnv",
    description=(
        "A production-grade OpenEnv environment simulating real-world multi-echelon "
        "supply chain disruption management. Three tasks (easy → hard) with "
        "dense partial-credit rewards and full typed Pydantic API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (thread-safe enough for single-user Space)
_env = SupplyChainEnv(seed=42)


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: Action


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Supply Chain OpenEnv</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Courier New',monospace;background:#0a0e1a;color:#c8d6f0;min-height:100vh;padding:40px 24px}
  .container{max-width:860px;margin:0 auto}
  h1{font-size:2rem;color:#4fc3f7;margin-bottom:8px;letter-spacing:2px}
  .sub{color:#7986cb;font-size:.85rem;margin-bottom:32px}
  .card{background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:20px;margin-bottom:16px}
  .card h2{color:#4fc3f7;font-size:1rem;margin-bottom:12px;letter-spacing:1px}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:bold;margin-right:6px}
  .easy{background:#1b5e20;color:#a5d6a7}.medium{background:#e65100;color:#ffcc80}.hard{background:#b71c1c;color:#ef9a9a}
  a{color:#4fc3f7;text-decoration:none}.a:hover{text-decoration:underline}
  pre{background:#0d1117;border:1px solid #1e3a5f;border-radius:6px;padding:16px;overflow-x:auto;font-size:.8rem;color:#a8c0d6;margin-top:12px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .link-btn{display:block;padding:10px 16px;background:#1e3a5f;border-radius:6px;color:#4fc3f7;text-align:center;margin-top:8px}
</style>
</head>
<body>
<div class="container">
  <h1>🏭 SUPPLY CHAIN OPENENV</h1>
  <p class="sub">Real-world multi-echelon supply chain disruption management · OpenEnv v1.0.0</p>

  <div class="card">
    <h2>📋 TASKS</h2>
    <p><span class="badge easy">EASY</span> <strong>task_1_easy</strong> — Single Supplier Failure · 20 steps · threshold 0.70</p>
    <p style="margin-top:8px"><span class="badge medium">MEDIUM</span> <strong>task_2_medium</strong> — Port Strike + Demand Spike · 25 steps · threshold 0.60</p>
    <p style="margin-top:8px"><span class="badge hard">HARD</span> <strong>task_3_hard</strong> — Cascading Global Failures · 30 steps · threshold 0.45</p>
  </div>

  <div class="card">
    <h2>⚡ QUICK START</h2>
    <pre># 1. Reset
curl -X POST /reset \\
  -H "Content-Type: application/json" \\
  -d '{"task_id":"task_1_easy","seed":42}'

# 2. Step
curl -X POST /step \\
  -H "Content-Type: application/json" \\
  -d '{"action":{"action_type":"do_nothing"}}'

# 3. Current state
curl /state</pre>
  </div>

  <div class="grid">
    <div class="card">
      <h2>🔗 LINKS</h2>
      <a class="link-btn" href="/docs">📖 Swagger UI</a>
      <a class="link-btn" href="/tasks">📋 Task List (JSON)</a>
      <a class="link-btn" href="/openenv.yaml">📄 openenv.yaml</a>
      <a class="link-btn" href="/health">💚 Health Check</a>
    </div>
    <div class="card">
      <h2>🎯 REWARD SIGNAL</h2>
      <p style="font-size:.85rem;line-height:1.7;color:#90a4ae">
        Dense per-step reward ∈ [0.0, 1.0]<br>
        Partial credit from step 1<br>
        Multi-component weighted scores<br>
        Penalises waste &amp; budget overrun<br>
        Grader breakdown in every response
      </p>
    </div>
  </div>
</div>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "env": "supply-chain-disruption"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_METADATA}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    """Reset the environment to the start of a task. Returns typed Observation dict."""
    try:
        obs = _env.reset(task_id=req.task_id, seed=req.seed)
        return obs.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    """Take one step with the given action. Returns StepResult."""
    try:
        result = _env.step(req.action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Return the current full environment state."""
    try:
        s = _env.state()
        return s.to_observation().to_dict()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    """Return the openenv.yaml specification file."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")
    try:
        with open(yaml_path) as f:
            return PlainTextResponse(f.read(), media_type="text/yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


@app.post("/validate")
def validate() -> Dict[str, Any]:
    """
    OpenEnv validate endpoint.
    Runs a mini smoke-test: reset + step on each task.
    """
    results = {}
    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        try:
            env = SupplyChainEnv(seed=0)
            obs = env.reset(task_id=task_id, seed=0)
            assert obs.step == 0
            result = env.step(Action(action_type=ActionType.DO_NOTHING))
            assert 0.0 <= result.reward <= 1.0
            state = env.state()
            assert state.step == 1
            results[task_id] = {"valid": True, "reward_sample": result.reward}
        except Exception as e:
            results[task_id] = {"valid": False, "error": str(e)}
    all_valid = all(v["valid"] for v in results.values())
    return {"valid": all_valid, "tasks": results}


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    """Interactive browser-based demo."""
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html")
    if os.path.exists(demo_path):
        with open(demo_path) as f:
            return f.read()
    return HTMLResponse("<h1>Demo not found</h1>", status_code=404)