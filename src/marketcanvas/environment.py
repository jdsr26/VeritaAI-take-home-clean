from __future__ import annotations

from enum import Enum
import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketcanvas.actions import CursorState, apply_low_level_action, apply_semantic_action
from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType
from marketcanvas.prompt_parser import ParsedPrompt, parse_prompt
from marketcanvas.renderer import render_to_array
from marketcanvas.reward import RewardBreakdown, compute_reward


_ACTION_TYPES = [
    "add_element", "move_element", "resize_element",
    "change_color", "change_text", "delete_element",
    "set_z_index", "noop",
]

_LOW_LEVEL_ACTION_TYPES = [
    "mouse_move", "mouse_click", "mouse_drag", "keyboard_type",
]


class ActionMode(str, Enum):
    SEMANTIC = "semantic"
    LOW_LEVEL = "low_level"


class MarketCanvasEnv(gym.Env):
    """Gymnasium environment for the MarketCanvas 2D design task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_steps: int = 50,
        canvas_width: int = 800,
        canvas_height: int = 600,
        action_mode: ActionMode | str = ActionMode.SEMANTIC,
        terminal_reward_only: bool = True,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.action_mode = ActionMode(action_mode)
        self.terminal_reward_only = terminal_reward_only

        self.canvas = Canvas(canvas_width, canvas_height)
        self.parsed_prompt = ParsedPrompt(raw="")
        self.step_count = 0
        self.cursor_state = CursorState()

        self.action_space = self._build_action_space(canvas_width, canvas_height)

        # Primary observation is the semantic state serialised as JSON;
        # visual RGB is secondary.  Agents that need a parsed dict can call
        # json.loads(obs["state_json"]).  The full dict is also in info["state"].
        self.observation_space = spaces.Dict({
            "state_json": spaces.Text(max_length=50000),
            "visual": spaces.Box(0, 255, shape=(canvas_height, canvas_width, 3), dtype=np.uint8),
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.canvas.clear()
        self.step_count = 0
        self.cursor_state = CursorState()

        prompt = (options or {}).get("prompt", "Create a marketing banner")
        self.parsed_prompt = parse_prompt(prompt)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self._dispatch(action)
        return self._build_step_result()

    def step_semantic(self, action_type: str, **params: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Convenience method for semantic actions with named parameters."""
        self.step_count += 1
        self._execute_semantic(action_type, params)
        return self._build_step_result()

    def step_low_level(self, action_type: str, **params: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Convenience method for low-level actions with named parameters."""
        self.step_count += 1
        self._execute_low_level(action_type, params)
        return self._build_step_result()

    def _build_step_result(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        reward_bd = compute_reward(self.canvas, self.parsed_prompt, self.step_count, self.max_steps)
        terminated = False
        truncated = self.step_count >= self.max_steps
        is_terminal = terminated or truncated

        # When terminal_reward_only is set, intermediate steps return 0.0 so
        # credit assignment is deferred to the end of the episode.
        reward = reward_bd.total if (not self.terminal_reward_only or is_terminal) else 0.0

        obs = self._get_obs()
        info = self._get_info()
        info["reward_breakdown"] = reward_bd.to_dict()
        info["is_terminal"] = is_terminal

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return render_to_array(self.canvas)
        return None

    def get_state(self) -> dict[str, Any]:
        return self.canvas.to_dict()

    def get_reward(self) -> RewardBreakdown:
        return compute_reward(self.canvas, self.parsed_prompt, self.step_count, self.max_steps)

    def _get_obs(self) -> dict[str, Any]:
        semantic_state = self.canvas.to_dict()
        return {
            "state_json": json.dumps(semantic_state, separators=(",", ":"), sort_keys=True),
            "visual": render_to_array(self.canvas),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "state": self.canvas.to_dict(),
            "step": self.step_count,
            "prompt": self.parsed_prompt.raw,
            "action_mode": self.action_mode.value,
        }

    def _build_action_space(self, canvas_width: int, canvas_height: int) -> spaces.Dict:
        if self.action_mode == ActionMode.SEMANTIC:
            return spaces.Dict({
                "action_type": spaces.Discrete(len(_ACTION_TYPES)),
                "id": spaces.Text(max_length=32),
                "type": spaces.Discrete(len(ElementType)),
                "content": spaces.Text(max_length=512),
                "x": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
                "y": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
                "new_x": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
                "new_y": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
                "width": spaces.Box(1, canvas_width, shape=(), dtype=np.int32),
                "height": spaces.Box(1, canvas_height, shape=(), dtype=np.int32),
                "new_width": spaces.Box(1, canvas_width, shape=(), dtype=np.int32),
                "new_height": spaces.Box(1, canvas_height, shape=(), dtype=np.int32),
                "color": spaces.Text(max_length=7),
                "text_color": spaces.Text(max_length=7),
                "hex_code": spaces.Text(max_length=7),
                "new_content": spaces.Text(max_length=512),
                "new_z": spaces.Box(0, 1000, shape=(), dtype=np.int32),
            })

        return spaces.Dict({
            "action_type": spaces.Discrete(len(_LOW_LEVEL_ACTION_TYPES)),
            "x": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
            "y": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
            "x1": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
            "y1": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
            "x2": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
            "y2": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
            "text": spaces.Text(max_length=256),
        })

    def _dispatch(self, action: dict[str, Any]) -> None:
        if self.action_mode == ActionMode.SEMANTIC:
            action_type = self._resolve_action_type(action, _ACTION_TYPES, default="noop")
            self._execute_semantic(action_type, action)
            return

        action_type = self._resolve_action_type(action, _LOW_LEVEL_ACTION_TYPES, default="mouse_move")
        self._execute_low_level(action_type, action)

    def _resolve_action_type(
        self,
        action: dict[str, Any],
        action_types: list[str],
        *,
        default: str,
    ) -> str:
        raw = action.get("action_type", default)
        if isinstance(raw, str):
            return raw if raw in action_types else default

        idx = int(raw)
        return action_types[idx] if 0 <= idx < len(action_types) else default

    def _execute_semantic(self, action_type: str, params: dict[str, Any]) -> None:
        apply_semantic_action(self.canvas, action_type, params)

    def _execute_low_level(self, action_type: str, params: dict[str, Any]) -> None:
        apply_low_level_action(self.canvas, self.cursor_state, action_type, params)
