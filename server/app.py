"""
PharmaOS — Drug Discovery Molecular Optimization RL Environment
server/app.py: FastAPI server with OpenEnv-compliant API + stunning Web UI.

Endpoints (OpenEnv Spec):
  POST /reset    — Start new episode (body: {task: str})
  POST /step     — Execute action   (body: {smiles: str, reasoning: str})
  GET  /state    — Episode metadata
  GET  /health   — Health check → 200 {status: healthy}
  GET  /tasks    — List tasks with descriptions
  WS   /ws       — Persistent WebSocket session (preferred)
  GET  /web      — Interactive debug UI (for human testing + judge demo)
  GET  /docs     — OpenAPI documentation

Built by: Team Fullstack Shinobi & Soumoditya Das
Event: Meta x PyTorch OpenEnv Hackathon 2026
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    PharmaAction, PharmaObservation, PharmaState,
    AVAILABLE_TASKS, TASK_DESCRIPTIONS, TASK_SUCCESS_THRESHOLDS,
)
from server.environment import PharmaEnvironment, IMPROVED_MOLECULE_HINTS
from server.agent import agent as pharma_agent
from server.compound_lookup import resolve_compound_query

_APP_NAME = "pharma-os"
_APP_VERSION = "2.0.0"
_APP_DESCRIPTION = (
    "Drug Discovery Molecular Optimization RL Environment for OpenEnv. "
    "Agents iteratively modify molecular SMILES strings and receive dense, "
    "deterministic rewards grounded in computational chemistry."
)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PharmaOS — Drug Discovery RL Environment",
    description=(
        "An OpenEnv-compatible Reinforcement Learning environment for "
        "AI-driven drug discovery and molecular optimization. "
        "An AI agent iteratively modifies molecular SMILES strings to "
        "improve drug-likeness using real computational chemistry (RDKit).\n\n"
        "Tasks:\n"
        "• lipinski_optimizer — Satisfy Lipinski Rule of Five (EASY)\n"
        "• qed_optimizer — Maximize QED drug-likeness score (MEDIUM)\n"
        "• multi_objective_designer — Multi-property + ADMET optimization (HARD)\n\n"
        "Built by Soumoditya Das for Meta × PyTorch OpenEnv Hackathon 2026."
    ),
    version=_APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_http_env: Optional[PharmaEnvironment] = None
_DEFAULT_TASK = os.getenv("PHARMAO_TASK", "lipinski_optimizer")


def get_http_env(task_name: Optional[str] = None) -> PharmaEnvironment:
    global _http_env, _DEFAULT_TASK
    if task_name is None:
        if _http_env is None:
            _http_env = PharmaEnvironment(task_name=_DEFAULT_TASK)
        return _http_env

    if _http_env is None or _http_env.task_name != task_name:
        _http_env = PharmaEnvironment(task_name=task_name)
    return _http_env


def build_agent_context(env: PharmaEnvironment) -> str:
    """Build agent context without assuming reset-only state fields exist."""
    state = env.get_state()
    current_smiles = getattr(env, "_current_smiles", "") or state.best_smiles or state.initial_smiles
    current_props = getattr(env, "_current_props", None)
    props_payload = (
        current_props.model_dump()
        if current_props is not None
        else {
            "best_score": state.best_score,
            "step_count": state.step_count,
            "visited_molecules": len(state.visited_molecules),
        }
    )
    task_name = state.task_name or env.task_name
    return (
        f"Current Molecule (SMILES): {current_smiles}\n"
        f"Properties: {props_payload}\n"
        f"Task: {task_name}"
    )


def _serialize_observation(obs: PharmaObservation) -> Dict[str, Any]:
    return obs.model_dump()


def _reset_response_payload(obs: PharmaObservation) -> Dict[str, Any]:
    observation = _serialize_observation(obs)
    return {
        **observation,
        "observation": observation,
        "reward": obs.reward,
        "done": obs.done,
    }


def _step_response_payload(
    obs: PharmaObservation,
    reward: float,
    done: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    observation = _serialize_observation(obs)
    return {
        **observation,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


def _state_response_payload(state: PharmaState) -> Dict[str, Any]:
    payload = state.model_dump()
    return {**payload, "state": payload}


def _extract_task_name(body: Dict[str, Any]) -> str:
    if not body:
        return _DEFAULT_TASK
    data = body.get("data")
    if isinstance(data, dict) and data.get("task"):
        return str(data["task"])
    return str(body.get("task", _DEFAULT_TASK))


def _extract_seed(body: Dict[str, Any]) -> Optional[int]:
    if not body:
        return None

    data = body.get("data")
    raw_seed = None
    if isinstance(data, dict) and "seed" in data:
        raw_seed = data.get("seed")
    elif "seed" in body:
        raw_seed = body.get("seed")

    if raw_seed in (None, ""):
        return None

    try:
        return int(raw_seed)
    except (TypeError, ValueError):
        return None


def _extract_action_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    action = body.get("action")
    if isinstance(action, dict):
        return action
    return body


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect to the stunning interactive Web UI."""
    return RedirectResponse(url="/web")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check — required by OpenEnv submission validator."""
    return {
        "status": "healthy",
        "environment": _APP_NAME,
        "name": _APP_NAME,
        "version": _APP_VERSION,
        "tasks": str(len(AVAILABLE_TASKS)),
    }


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """OpenEnv runtime metadata endpoint."""
    return {
        "name": _APP_NAME,
        "description": _APP_DESCRIPTION,
        "version": _APP_VERSION,
        "author": "Soumoditya Das",
        "documentation_url": "/docs",
        "task_count": len(AVAILABLE_TASKS),
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Return JSON schemas for the action, observation, and state models."""
    return {
        "action": PharmaAction.model_json_schema(),
        "observation": PharmaObservation.model_json_schema(),
        "state": PharmaState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Minimal JSON-RPC endpoint for OpenEnv runtime validation."""
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}}

    if method == "openenv/session/create":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"session_id": f"http-{uuid.uuid4().hex[:12]}"},
        }

    if method == "openenv/session/close":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"session_id": params.get("session_id"), "closed": True},
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32600,
            "message": "Unsupported JSON-RPC request for PharmaOS MCP endpoint",
        },
    }


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": AVAILABLE_TASKS,
        "descriptions": TASK_DESCRIPTIONS,
        "success_thresholds": TASK_SUCCESS_THRESHOLDS,
        "hints": {t: IMPROVED_MOLECULE_HINTS.get(t, []) for t in AVAILABLE_TASKS},
    }


@app.post("/reset")
async def reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """
    Reset the environment and start a new episode.
    Body (optional): {"task": "lipinski_optimizer" | "qed_optimizer" | "multi_objective_designer"}
    """
    task_name = _extract_task_name(body)
    seed = _extract_seed(body)
    env = get_http_env(task_name)
    obs = env.reset(seed=seed)
    return _reset_response_payload(obs)


@app.post("/step")
async def step(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """
    Execute one environment step.
    Supports both direct and OpenEnv-style action payloads.
    """
    env = get_http_env()
    action_payload = _extract_action_payload(body)
    action = PharmaAction(
        smiles=action_payload.get("smiles", ""),
        reasoning=action_payload.get("reasoning", ""),
        metadata=action_payload.get("metadata", {}),
    )
    obs, reward, done, info = env.step(action)
    return _step_response_payload(obs, reward, done, info)


@app.get("/state")
async def state(task: Optional[str] = None) -> Dict[str, Any]:
    """Return current episode state metadata."""
    env = get_http_env(task)
    return _state_response_payload(env.get_state())


@app.post("/api/chat")
async def chat(body: Dict[str, Any]) -> Dict[str, Any]:
    """Provide AI assistant reasoning via natural language."""
    query = body.get("query", "")
    context = body.get("context", "")
    
    if not context:
        env = get_http_env()
        context = build_agent_context(env)

    if not query:
        return {"response": "Please provide a valid query."}
    return {"response": pharma_agent.generate_response(query, context)}

@app.post("/api/reasoning_trace")
async def reasoning_trace(body: Dict[str, Any]) -> Dict[str, Any]:
    """Provide structured ChainOfThought reasoning trace."""
    query = body.get("query", "")
    context = body.get("context", "")
    
    if not context:
        env = get_http_env()
        context = build_agent_context(env)

    if not query:
        return {"error": "Please provide a valid query."}
    trace = pharma_agent.get_reasoning_trace(query, context)
    return trace.to_dict()


@app.get("/api/compound_lookup")
async def compound_lookup(q: str) -> Dict[str, Any]:
    """Resolve exact compounds, aliases, and broad product classes into validated compounds."""
    return resolve_compound_query(q)


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

async def _process_ws_message(websocket: WebSocket, env, msg: dict):
    msg_type = msg.get("type", "")
    if msg_type == "reset":
        task_name = msg.get("task", _DEFAULT_TASK)
        if task_name not in AVAILABLE_TASKS:
            await websocket.send_json({"error": f"Unknown task: {task_name}"})
            return env
        new_env = PharmaEnvironment(task_name=task_name)
        obs = new_env.reset()
        await websocket.send_json(obs.model_dump())
        return new_env

    if msg_type == "step":
        action = PharmaAction(
            smiles=msg.get("smiles", ""),
            reasoning=msg.get("reasoning", ""),
        )
        obs, reward, done, info = env.step(action)
        await websocket.send_json({**obs.model_dump(), "reward": reward, "done": done, "info": info})
        return env

    if msg_type == "state":
        await websocket.send_json(env.get_state().model_dump())
        return env

    if msg_type == "chat":
        query = msg.get("query", "")
        context = msg.get("context", "")
        if not context:
            context = build_agent_context(env)
            
        trace = pharma_agent.get_reasoning_trace(query, context)
        prefix = f"🧠 [Layer: {trace.level}] "
        if trace.formula and trace.formula not in ["Fallback Heuristic", "SMILES Grammar", "LLM Generative Priority"]:
            prefix += f"[Eq: {trace.formula}] "
        response = f"{prefix}\n\n{trace.recommendation}"
        
        await websocket.send_json({"type": "chat_response", "response": response, "trace": trace.to_dict()})
        return env

    await websocket.send_json({"error": f"Unknown type '{msg_type}'. Use: reset, step, state, chat"})
    return env


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session. Each connection gets its own isolated environment.
    """
    await websocket.accept()
    env = PharmaEnvironment(task_name=_DEFAULT_TASK)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            env = await _process_ws_message(websocket, env, msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ─── Web UI ───────────────────────────────────────────────────────────────────


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Premium 3D interactive Web UI — loads from server/web_ui.html."""
    import pathlib
    ui_path = pathlib.Path(__file__).parent / "web_ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UI not found — run server from pharma-os/</h1>", status_code=500)



WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PharmaOS — Drug Discovery RL Environment</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080b14;
    --bg2: #0d1220;
    --bg3: #111827;
    --card: #131c2e;
    --card2: #1a2540;
    --border: #1e2d4a;
    --border2: #243557;
    --purple: #7c3aed;
    --purple2: #9333ea;
    --violet: #6d28d9;
    --teal: #06b6d4;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --text3: #64748b;
    --glow: rgba(124,58,237,0.15);
    --glow2: rgba(6,182,212,0.10);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background grid */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Header ─────────────────────────────────── */
  .header {
    position: sticky;
    top: 0;
    background: rgba(13, 18, 32, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    z-index: 100;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
  }
  .logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--purple), var(--teal));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 24px var(--glow), inset 0 0 10px rgba(255,255,255,0.2);
    flex-shrink: 0;
    animation: float 4s ease-in-out infinite;
  }
  @keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
  }
  .header-text h1 {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }
  .header-text p { color: var(--text2); font-size: 0.8rem; margin-top: 2px; }
  .header-badges { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
  .badge {
    padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600;
    border: 1px solid;
  }
  .badge-purple { background: rgba(124,58,237,0.15); border-color: rgba(124,58,237,0.4); color: #a78bfa; }
  .badge-teal   { background: rgba(6,182,212,0.12);  border-color: rgba(6,182,212,0.4);  color: #67e8f9; }
  .badge-green  { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.4); color: #6ee7b7; }

  /* ── Layout ──────────────────────────────────── */
  .app { position: relative; z-index: 1; }
  .top-bar {
    display: flex; gap: 12px; padding: 16px 32px;
    background: rgba(13, 18, 32, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    align-items: center; flex-wrap: wrap; z-index: 50; position: relative;
  }
  .task-pill {
    padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    cursor: pointer; border: 1.5px solid var(--border2); background: var(--card);
    color: var(--text2); transition: all 0.2s; user-select: none;
  }
  .task-pill:hover { border-color: var(--purple); color: #a78bfa; }
  .task-pill.active { background: rgba(124,58,237,0.2); border-color: var(--purple); color: #a78bfa; }
  .task-pill.easy   { border-color: #166534; color: #6ee7b7; }
  .task-pill.easy.active { background: rgba(16,185,129,0.15); border-color: var(--green); color: var(--green); }
  .task-pill.medium { border-color: #92400e; color: #fcd34d; }
  .task-pill.medium.active { background: rgba(245,158,11,0.15); border-color: var(--amber); color: var(--amber); }
  .task-pill.hard   { border-color: #7f1d1d; color: #fca5a5; }
  .task-pill.hard.active { background: rgba(239,68,68,0.15); border-color: var(--red); color: var(--red); }
  .reset-btn {
    margin-left: auto;
    padding: 8px 22px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
    background: linear-gradient(135deg, var(--purple), var(--violet));
    color: white; border: none; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 0 15px var(--glow);
  }
  .reset-btn:hover { opacity: 0.88; transform: translateY(-1px); }

  /* ── Main Grid ───────────────────────────────── */
  .main {
    display: grid;
    grid-template-columns: 340px 1fr;
    grid-template-rows: auto auto;
    gap: 0;
    min-height: calc(100vh - 120px);
  }
  .panel { background: var(--card); border-right: 1px solid var(--border); }
  .panel-right { background: var(--bg); border-left: none; }

  /* ── Left Panel ───────────────────────────────── */
  .left-panel { 
    display: flex; flex-direction: column; gap: 0;
    border-right: 1px solid var(--border);
  }
  .section { padding: 16px 20px; border-bottom: 1px solid var(--border); }
  .section-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: var(--text3); margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
  }

  /* Score display */
  .score-hero {
    display: flex; flex-direction: column; align-items: center;
    padding: 20px; gap: 8px;
    background: linear-gradient(180deg, rgba(124,58,237,0.05) 0%, transparent 100%);
  }
  .score-num {
    font-size: 3rem; font-weight: 800; letter-spacing: -2px;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .score-label { font-size: 0.75rem; color: var(--text3); font-weight: 500; }
  .progress-track {
    width: 100%; height: 6px; background: var(--bg3); border-radius: 3px; margin-top: 4px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
    background: linear-gradient(90deg, var(--purple), var(--teal));
  }

  /* Molecule viewer */
  .mol-viewer {
    background: rgba(17, 24, 39, 0.6); 
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.05); 
    border-radius: 12px;
    padding: 8px; min-height: 200px; display: flex; align-items: center; justify-content: center;
    position: relative; overflow: hidden;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
  }
  .mol-viewer svg { max-width: 100%; max-height: 190px; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.4)); transform: scale(1.05); transition: transform 0.3s ease; }
  .mol-viewer:hover svg { transform: scale(1.1); }
  .mol-viewer .mol-placeholder {
    color: var(--text3); font-size: 0.8rem; text-align: center; line-height: 1.6;
  }
  .smiles-chip {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--teal); word-break: break-all; line-height: 1.5;
    cursor: pointer; transition: border-color 0.2s;
  }
  .smiles-chip:hover { border-color: var(--purple); }
  .copy-hint { font-size: 0.65rem; color: var(--text3); margin-top: 4px; }

  /* Property rows */
  .prop-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .prop-item {
    background: var(--bg3); border-radius: 8px; padding: 8px 10px;
    border: 1px solid var(--border);
  }
  .prop-key { font-size: 0.65rem; color: var(--text3); font-weight: 600; letter-spacing: 0.5px; }
  .prop-val {
    font-size: 0.85rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
  }
  .prop-val.good { color: var(--green); }
  .prop-val.warn { color: var(--amber); }
  .prop-val.bad  { color: var(--red); }
  .prop-val.neutral { color: var(--text); }

  /* ADMET section */
  .admet-row {
    display: flex; align-items: center; gap: 8px; padding: 5px 0;
    border-bottom: 1px solid var(--border);
  }
  .admet-row:last-child { border-bottom: none; }
  .admet-label { font-size: 0.72rem; color: var(--text2); flex: 1; }
  .admet-val { font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .pill {
    padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 700;
  }
  .pill-green { background: rgba(16,185,129,0.2); color: #6ee7b7; }
  .pill-amber { background: rgba(245,158,11,0.2); color: #fcd34d; }
  .pill-red   { background: rgba(239,68,68,0.2);  color: #fca5a5; }
  .pill-grey  { background: rgba(100,116,139,0.2); color: #94a3b8; }

  /* ── Right Panel (Input + Output) ─────────────── */
  .right-panel { display: flex; flex-direction: column; }
  .input-zone {
    padding: 20px 24px;
    display: flex; gap: 10px; align-items: flex-start;
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
    flex-wrap: wrap;
  }
  .smiles-field {
    flex: 1; min-width: 200px;
    background: var(--bg3); border: 1.5px solid var(--border2);
    border-radius: 10px; color: var(--text);
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    padding: 10px 14px; outline: none; transition: border-color 0.2s;
  }
  .smiles-field:focus { border-color: var(--purple); box-shadow: 0 0 0 3px rgba(124,58,237,0.1); }
  .reasoning-field {
    flex: 1; min-width: 150px;
    background: var(--bg3); border: 1.5px solid var(--border2);
    border-radius: 10px; color: var(--text2);
    font-family: 'Inter', sans-serif; font-size: 0.8rem;
    padding: 10px 14px; outline: none; transition: border-color 0.2s;
  }
  .reasoning-field:focus { border-color: var(--teal); }
  .step-btn {
    padding: 10px 28px; border-radius: 10px; font-size: 0.9rem; font-weight: 700;
    background: linear-gradient(135deg, var(--teal), var(--purple));
    color: white; border: none; cursor: pointer; transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3), inset 0 1px 0 rgba(255,255,255,0.2); 
    white-space: nowrap;
  }
  .step-btn:hover { opacity: 1; transform: translateY(-2px) scale(1.02); box-shadow: 0 0 30px rgba(124, 58, 237, 0.5); }
  .step-btn:active { transform: translateY(1px); }
  .step-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; filter: grayscale(100%); }

  /* Hint pills */
  .hint-strip {
    padding: 8px 24px; background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  }
  .hint-label { font-size: 0.68rem; color: var(--text3); font-weight: 600; margin-right: 4px; }
  .hint-mol {
    padding: 3px 10px; border-radius: 12px; font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.2);
    color: var(--teal); cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }
  .hint-mol:hover { background: rgba(6,182,212,0.18); border-color: var(--teal); }

  /* Output log */
  .output-log {
    flex: 1; overflow-y: auto; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 8px;
    max-height: calc(100vh - 280px);
  }
  .log-entry {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.77rem;
    color: #a0f0c0; white-space: pre-wrap; line-height: 1.6;
    animation: slideIn 0.2s ease;
  }
  .log-entry.error-entry { color: #fca5a5; border-color: rgba(239,68,68,0.3); }
  .log-entry.reset-entry { color: #a78bfa; border-color: rgba(124,58,237,0.3); }
  @keyframes slideIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

  /* History table */
  .history-section {
    padding: 16px 24px; border-top: 1px solid var(--border);
    background: var(--bg2);
  }
  .history-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  .history-table th {
    text-align: left; padding: 6px 8px; color: var(--text3);
    font-weight: 600; letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
  }
  .history-table td {
    padding: 6px 8px; border-bottom: 1px solid rgba(30,45,74,0.5);
    font-family: 'JetBrains Mono', monospace; color: var(--text2);
  }
  .history-table tr:hover td { background: var(--card); cursor: pointer; }
  .score-td { font-weight: 700; }

  /* Step counter */
  .step-info {
    display: flex; gap: 16px; align-items: center;
    padding: 8px 24px; background: var(--bg2);
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem; color: var(--text2);
  }
  .step-info span { font-family: 'JetBrains Mono', monospace; color: var(--text); font-weight: 600; }

  /* Done banner */
  .done-banner {
    padding: 10px 24px; text-align: center;
    background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(6,182,212,0.1), rgba(124,58,237,0.15));
    border-bottom: 1px solid rgba(124,58,237,0.3);
    font-size: 0.85rem; font-weight: 600; color: #a78bfa;
    animation: glow-pulse 2s ease infinite;
  }
  @keyframes glow-pulse {
    0%,100% { box-shadow: 0 0 20px rgba(124,58,237,0.1); }
    50%      { box-shadow: 0 0 40px rgba(124,58,237,0.25); }
  }

  /* Spinner */
  .spinner {
    width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.2);
    border-top-color: white; border-radius: 50%;
    animation: spin 0.7s linear infinite; display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--purple); }

  /* Responsive */
  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
    .left-panel { border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>
<div class="header">
  <div class="logo">🧬</div>
  <div class="header-text">
    <h1>PharmaOS</h1>
    <p>Drug Discovery Molecular Optimization RL Environment · Meta × PyTorch Hackathon 2026</p>
  </div>
  <div class="header-badges">
    <span class="badge badge-purple">OpenEnv Compliant</span>
    <span class="badge badge-teal">RDKit v2024</span>
    <span class="badge badge-green">3 Tasks</span>
  </div>
</div>

<div class="app">
  <div class="top-bar">
    <span class="task-pill easy active" id="pill-lipinski" onclick="selectTask('lipinski_optimizer')">🟢 Lipinski Optimizer</span>
    <span class="task-pill medium" id="pill-qed" onclick="selectTask('qed_optimizer')">🟡 QED Optimizer</span>
    <span class="task-pill hard" id="pill-multi" onclick="selectTask('multi_objective_designer')">🔴 Multi-Parameter MPO</span>
    <button class="reset-btn" onclick="resetEnv()">⟳ New Episode</button>
  </div>

  <div class="main">
    <!-- LEFT: Properties Panel -->
    <div class="left-panel">
      <!-- Score hero -->
      <div class="score-hero section">
        <div class="score-num" id="score-num">0.000</div>
        <div class="score-label" id="score-label">Best Score / 1.0</div>
        <div style="width:100%">
          <div class="progress-track"><div class="progress-fill" id="prog" style="width:0%"></div></div>
        </div>
        <div style="font-size:0.72rem;color:var(--text3);margin-top:4px;" id="step-badge">Step 0 / —</div>
      </div>

      <!-- Molecule viewer -->
      <div class="section">
        <div class="section-title">🔬 Current Molecule</div>
        <div class="mol-viewer" id="mol-viewer">
          <div class="mol-placeholder">Click Reset to start<br>an episode</div>
        </div>
        <div class="smiles-chip" id="smiles-display" onclick="copySMILES()" title="Click to copy">
          —
        </div>
        <div class="copy-hint">Click SMILES to copy</div>
      </div>

      <!-- Physico-chemical properties -->
      <div class="section">
        <div class="section-title">📊 Molecular Properties</div>
        <div class="prop-grid" id="props-grid">
          <!-- filled by JS -->
        </div>
      </div>

      <!-- ADMET -->
      <div class="section">
        <div class="section-title">🧪 ADMET Profile</div>
        <div id="admet-section">
          <div style="color:var(--text3);font-size:0.8rem;">No data yet.</div>
        </div>
      </div>
    </div>

    <!-- RIGHT: I/O Panel -->
    <div class="right-panel">
      <div id="done-banner" style="display:none;" class="done-banner">
        🏁 Episode Complete — Reset for a new challenge!
      </div>

      <div class="step-info">
        Task: <span id="si-task">lipinski_optimizer</span>
        &nbsp;|&nbsp; Step: <span id="si-step">0</span>
        &nbsp;|&nbsp; Unique Mols: <span id="si-mols">0</span>
        &nbsp;|&nbsp; Scaffolds: <span id="si-scaf">0</span>
        &nbsp;|&nbsp; PAINS: <span id="si-pains">0</span>
      </div>

      <!-- Input zone -->
      <div class="input-zone">
        <input class="smiles-field" id="smiles-input" placeholder="Enter SMILES (e.g. CC(=O)Nc1ccc(cc1)O)" value="CC(=O)Nc1ccc(cc1)O" />
        <input class="reasoning-field" id="reasoning-input" placeholder="Agent reasoning (optional)" />
        <button class="step-btn" id="step-btn" onclick="takeStep()">▶ Step</button>
      </div>

      <!-- Hint strip -->
      <div class="hint-strip" id="hint-strip">
        <span class="hint-label">💡 HINTS:</span>
      </div>

      <!-- Output log -->
      <div class="output-log" id="output-log">
        <div style="color:var(--text3);font-size:0.8rem;padding:20px;text-align:center;">
          Click "New Episode" to start.<br>
          The agent will receive molecular properties and ADMET feedback after each step.
        </div>
      </div>

      <!-- History table -->
      <div class="history-section">
        <div class="section-title" style="margin-bottom:8px;">📈 Episode History</div>
        <table class="history-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>SMILES</th>
              <th>Score</th>
              <th>QED</th>
              <th>SA</th>
            </tr>
          </thead>
          <tbody id="history-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const BASE = "";
let current_task = "lipinski_optimizer";
let episode_done = false;
let last_obs = null;
const HINTS = {};

// Fetch hints on load
fetch(BASE + "/tasks").then(r => r.json()).then(d => {
  Object.assign(HINTS, d.hints || {});
  updateHints();
}).catch(() => {});

function selectTask(task) {
  current_task = task;
  document.querySelectorAll('.task-pill').forEach(p => p.classList.remove('active'));
  const pillMap = {
    'lipinski_optimizer': 'pill-lipinski',
    'qed_optimizer': 'pill-qed',
    'multi_objective_designer': 'pill-multi'
  };
  const pill = document.getElementById(pillMap[task]);
  if (pill) pill.classList.add('active');
  document.getElementById('si-task').textContent = task;
  updateHints();
}

function updateHints() {
  const strip = document.getElementById('hint-strip');
  const mols = HINTS[current_task] || [];
  strip.innerHTML = '<span class="hint-label">💡 HINTS:</span>';
  mols.slice(0, 5).forEach(m => {
    const span = document.createElement('span');
    span.className = 'hint-mol';
    span.textContent = m.length > 30 ? m.slice(0, 28) + '…' : m;
    span.title = m;
    span.onclick = () => { document.getElementById('smiles-input').value = m; };
    strip.appendChild(span);
  });
}

function log(text, cls = '') {
  const el = document.getElementById('output-log');
  const div = document.createElement('div');
  div.className = 'log-entry ' + cls;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function clearLog() {
  document.getElementById('output-log').innerHTML = '';
  document.getElementById('history-tbody').innerHTML = '';
}

function updateScore(score, step, max_steps) {
  const s = parseFloat(score) || 0;
  document.getElementById('score-num').textContent = s.toFixed(3);
  const pct = Math.min(100, s * 100);
  document.getElementById('prog').style.width = pct + '%';
  document.getElementById('step-badge').textContent = `Step ${step} / ${max_steps || '—'}`;
}

function updateSMILES(smiles) {
  document.getElementById('smiles-display').textContent = smiles || '—';
  document.getElementById('smiles-input').value = smiles || '';
}

function updateMolViewer(obs) {
  const viewer = document.getElementById('mol-viewer');
  if (obs.mol_svg) {
    viewer.innerHTML = obs.mol_svg;
  } else {
    viewer.innerHTML = '<div class="mol-placeholder">SVG unavailable<br><small>' + (obs.current_smiles||'').slice(0,40) + '</small></div>';
  }
}

function updateProps(props) {
  if (!props) return;
  const grid = document.getElementById('props-grid');
  const fields = [
    { key: 'MW', val: (props.molecular_weight||'—') + ' Da', cls: props.molecular_weight < 500 ? 'good' : 'bad' },
    { key: 'LogP', val: props.logp != null ? props.logp.toFixed(3) : '—', cls: (props.logp != null && props.logp < 5) ? 'good' : 'bad' },
    { key: 'HBD', val: props.hbd != null ? props.hbd : '—', cls: (props.hbd != null && props.hbd <= 5) ? 'good' : 'warn' },
    { key: 'HBA', val: props.hba != null ? props.hba : '—', cls: (props.hba != null && props.hba <= 10) ? 'good' : 'warn' },
    { key: 'QED', val: props.qed != null ? props.qed.toFixed(4) : '—', cls: props.qed > 0.6 ? 'good' : props.qed > 0.3 ? 'warn' : 'bad' },
    { key: 'SA Score', val: props.sa_score != null ? props.sa_score.toFixed(2) : '—', cls: props.sa_score < 3 ? 'good' : props.sa_score < 6 ? 'warn' : 'bad' },
    { key: 'TPSA', val: props.tpsa != null ? props.tpsa.toFixed(1) + ' Å²' : '—', cls: (props.tpsa != null && props.tpsa < 140) ? 'good' : 'warn' },
    { key: 'Fsp³', val: props.fsp3 != null ? props.fsp3.toFixed(3) : '—', cls: props.fsp3 > 0.3 ? 'good' : 'warn' },
    { key: 'RotBonds', val: props.rotatable_bonds != null ? props.rotatable_bonds : '—', cls: (props.rotatable_bonds != null && props.rotatable_bonds < 10) ? 'good' : 'warn' },
    { key: 'Violations', val: (props.lipinski_violations||0) + '/4', cls: props.lipinski_violations === 0 ? 'good' : props.lipinski_violations <= 1 ? 'warn' : 'bad' },
    { key: 'Tanimoto', val: props.fingerprint_similarity != null ? props.fingerprint_similarity.toFixed(4) : '—', cls: 'neutral' },
    { key: 'LigEff', val: props.ligand_efficiency != null ? props.ligand_efficiency.toFixed(4) : '—', cls: 'neutral' },
  ];
  grid.innerHTML = fields.map(f =>
    `<div class="prop-item"><div class="prop-key">${f.key}</div><div class="prop-val ${f.cls}">${f.val}</div></div>`
  ).join('');
}

function updateADMET(props, admet) {
  if (!props) return;
  const sec = document.getElementById('admet-section');
  const logS = props.logS != null ? props.logS.toFixed(2) : '—';
  const solClass = admet?.solubility_class || '—';
  const bbb = props.bbb_score != null ? props.bbb_score.toFixed(3) : '—';
  const herg = props.herg_risk != null ? props.herg_risk.toFixed(3) : '—';
  const hergCls = admet?.herg_class || '—';
  const pains = props.pains_alert;
  const oral = admet?.oral_bioavailable;
  const ml_conf = props.ml_bioactivity_confidence != null ? (props.ml_bioactivity_confidence * 100).toFixed(1) + "%" : "—";
  const mlPill = props.ml_bioactivity_confidence != null && props.ml_bioactivity_confidence > 0.7 ? 'pill-green' : 'pill-amber';

  const hergPill = hergCls === 'Low' ? 'pill-green' : hergCls === 'Medium' ? 'pill-amber' : 'pill-red';
  const solPill = solClass === 'High' ? 'pill-green' : solClass === 'Moderate' ? 'pill-amber' : 'pill-red';
  const bbbPill = admet?.bbb_penetrant ? 'pill-green' : 'pill-grey';

  sec.innerHTML = `
    <div class="admet-row"><span class="admet-label">Deep Learning Confidence</span><span class="admet-val">${ml_conf}</span><span class="pill ${mlPill}">PyTorch Net</span></div>
    <div class="admet-row"><span class="admet-label">Solubility (LogS)</span><span class="admet-val">${logS}</span><span class="pill ${solPill}">${solClass}</span></div>
    <div class="admet-row"><span class="admet-label">BBB Score</span><span class="admet-val">${bbb}</span><span class="pill ${bbbPill}">${admet?.bbb_penetrant ? 'CNS-active' : 'CNS-excluded'}</span></div>
    <div class="admet-row"><span class="admet-label">hERG Risk</span><span class="admet-val">${herg}</span><span class="pill ${hergPill}">${hergCls}</span></div>
    <div class="admet-row"><span class="admet-label">PAINS Alert</span><span></span><span class="pill ${pains ? 'pill-red' : 'pill-green'}">${pains ? '⚠ HIT' : '✓ CLEAN'}</span></div>
    <div class="admet-row"><span class="admet-label">Oral Bioavail.</span><span></span><span class="pill ${oral ? 'pill-green' : 'pill-red'}">${oral ? 'YES' : 'NO'}</span></div>
  `;
}

function addHistoryRow(obs, step) {
  const tbody = document.getElementById('history-tbody');
  const p = obs.properties || {};
  const row = tbody.insertRow(0);
  row.innerHTML = `
    <td>${step}</td>
    <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;font-size:0.68rem;" title="${obs.current_smiles}">${(obs.current_smiles||'').slice(0,35)}…</td>
    <td class="score-td" style="color:${(obs.best_score||0)>0.6?'#6ee7b7':(obs.best_score||0)>0.3?'#fcd34d':'#fca5a5'}">${(obs.best_score||0).toFixed(4)}</td>
    <td>${p.qed!=null?p.qed.toFixed(3):'—'}</td>
    <td>${p.sa_score!=null?p.sa_score.toFixed(2):'—'}</td>
  `;
  row.onclick = () => { document.getElementById('smiles-input').value = obs.current_smiles; };
}

function updateInfo(obs) {
  document.getElementById('si-step').textContent = obs.step_count || 0;
  document.getElementById('si-mols').textContent = obs.visited_count || 0;
}

function applyObs(obs, isReset = false) {
  last_obs = obs;
  episode_done = obs.done;
  const step = obs.step_count || 0;

  updateScore(obs.best_score, step, null);
  updateSMILES(obs.current_smiles);
  updateMolViewer(obs);
  updateProps(obs.properties);
  updateADMET(obs.properties, obs.admet);
  updateInfo(obs);

  if (!isReset && !obs.done) {
    addHistoryRow(obs, step);
  }

  const doneBanner = document.getElementById('done-banner');
  if (obs.done) {
    doneBanner.style.display = 'block';
    doneBanner.textContent = `🏁 Episode Complete! Best Score: ${(obs.best_score||0).toFixed(4)} — Reset for a new challenge!`;
    document.getElementById('step-btn').disabled = true;
  } else {
    doneBanner.style.display = 'none';
    document.getElementById('step-btn').disabled = false;
  }

  log(obs.feedback || JSON.stringify(obs, null, 2), isReset ? 'reset-entry' : '');
}

async function resetEnv() {
  clearLog();
  episode_done = false;
  document.getElementById('step-btn').disabled = false;
  document.getElementById('done-banner').style.display = 'none';
  document.getElementById('si-scaf').textContent = '0';
  document.getElementById('si-pains').textContent = '0';

  const btn = document.querySelector('.reset-btn');
  btn.innerHTML = '<div class="spinner"></div> Resetting…';
  btn.disabled = true;

  try {
    const r = await fetch(BASE + "/reset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task: current_task})
    });
    const obs = await r.json();
    applyObs(obs, true);
  } catch(e) {
    log("❌ Error: " + e.message, 'error-entry');
  } finally {
    btn.innerHTML = '⟳ New Episode';
    btn.disabled = false;
  }
}

async function takeStep() {
  if (episode_done) { log("Episode done — reset first.", 'error-entry'); return; }
  const smiles = document.getElementById('smiles-input').value.trim();
  const reasoning = document.getElementById('reasoning-input').value.trim();
  if (!smiles) { log("⚠️ Enter a SMILES string first.", 'error-entry'); return; }

  const btn = document.getElementById('step-btn');
  btn.innerHTML = '<div class="spinner"></div>';
  btn.disabled = true;

  try {
    const r = await fetch(BASE + "/step", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({smiles, reasoning})
    });
    const d = await r.json();
    applyObs(d, false);

    // Update scaffold/pains from state
    fetch(BASE + "/state").then(r2 => r2.json()).then(st => {
      document.getElementById('si-scaf').textContent = st.unique_scaffolds || 0;
      document.getElementById('si-pains').textContent = st.pains_attempts || 0;
    }).catch(() => {});
  } catch(e) {
    log("❌ Error: " + e.message, 'error-entry');
  } finally {
    btn.innerHTML = '▶ Step';
    btn.disabled = false;
  }
}

function copySMILES() {
  const s = document.getElementById('smiles-display').textContent;
  if (s && s !== '—') {
    navigator.clipboard.writeText(s).then(() => {
      const el = document.querySelector('.copy-hint');
      el.textContent = '✓ Copied!';
      setTimeout(() => { el.textContent = 'Click SMILES to copy'; }, 1500);
    });
  }
}

// Auto-reset on load
window.addEventListener('load', () => {
  resetEnv();
  updateHints();
});
</script>

<div style="text-align: center; margin-top: 30px; font-size: 0.85rem; color: var(--text3); border-top: 1px solid var(--border); padding-top: 20px;">
  Built with ❤️ by <strong>Team Fullstack Shinobi</strong> and <strong>Soumoditya Das</strong><br>
  Meta x PyTorch OpenEnv Hackathon 2026
</div>

</body>
</html>"""


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main(host: Optional[str] = None, port: Optional[int] = None, workers: Optional[int] = None):
    """Run the PharmaOS server locally or inside a container."""
    resolved_host = host or os.getenv("HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("PORT", 7860))
    resolved_workers = workers or int(os.getenv("WORKERS", 1))
    uvicorn.run("server.app:app", host=resolved_host, port=resolved_port, workers=resolved_workers)


if __name__ == "__main__":
    main()
