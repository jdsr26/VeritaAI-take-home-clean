from __future__ import annotations

import uuid
from typing import Any

from fastmcp import FastMCP

from marketcanvas.actions import apply_semantic_action
from marketcanvas.canvas import Canvas
from marketcanvas.prompt_parser import parse_prompt, ParsedPrompt
from marketcanvas.renderer import render_to_base64
from marketcanvas.reward import compute_reward

mcp = FastMCP("MarketCanvas")

# ---------------------------------------------------------------------------
# Session management — each caller can work with an isolated canvas.
# A default session ("default") provides backward-compatible single-client
# usage; callers that need isolation pass an explicit session_id.
# ---------------------------------------------------------------------------

_DEFAULT_SESSION = "default"


class _Session:
    __slots__ = ("canvas", "parsed_prompt", "step_count", "max_steps")

    def __init__(self, max_steps: int = 50) -> None:
        self.canvas = Canvas()
        self.parsed_prompt = ParsedPrompt(raw="")
        self.step_count = 0
        self.max_steps = max_steps


_sessions: dict[str, _Session] = {}


def _get_session(session_id: str | None) -> _Session:
    sid = session_id or _DEFAULT_SESSION
    if sid not in _sessions:
        _sessions[sid] = _Session()
    return _sessions[sid]


@mcp.tool
def reset_environment(prompt: str, session_id: str | None = None) -> dict[str, Any]:
    """Reset canvas and set a new target prompt.

    Pass *session_id* to operate on an isolated session.  Omit it (or pass
    ``None``) to use the shared default session.  Returns new session_id so
    callers can track it.
    """
    sid = session_id or str(uuid.uuid4())
    sess = _Session()
    sess.parsed_prompt = parse_prompt(prompt)
    _sessions[sid] = sess
    return {"state": sess.canvas.to_dict(), "prompt": prompt, "session_id": sid}


@mcp.tool
def get_canvas_state(session_id: str | None = None) -> dict[str, Any]:
    """Returns semantic JSON state of the canvas."""
    sess = _get_session(session_id)
    return sess.canvas.to_dict()


@mcp.tool
def execute_action(action_type: str, params: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    """Execute an action on the canvas. Returns new state, reward, and done flag."""
    sess = _get_session(session_id)
    sess.step_count += 1
    apply_semantic_action(sess.canvas, action_type, params)

    reward = compute_reward(sess.canvas, sess.parsed_prompt, sess.step_count, sess.max_steps)
    done = sess.step_count >= sess.max_steps

    return {
        "state": sess.canvas.to_dict(),
        "reward": reward.to_dict(),
        "done": done,
        "step": sess.step_count,
        "session_id": session_id or _DEFAULT_SESSION,
    }


@mcp.tool
def get_current_reward(session_id: str | None = None) -> dict[str, Any]:
    """Calculate and return current reward without stepping."""
    sess = _get_session(session_id)
    reward = compute_reward(sess.canvas, sess.parsed_prompt, sess.step_count, sess.max_steps)
    return reward.to_dict()


@mcp.tool
def render_canvas(session_id: str | None = None) -> str:
    """Returns base64-encoded PNG of current canvas."""
    sess = _get_session(session_id)
    return render_to_base64(sess.canvas)


if __name__ == "__main__":
    mcp.run()
