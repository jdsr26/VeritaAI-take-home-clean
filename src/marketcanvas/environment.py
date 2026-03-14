from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketcanvas.actions import apply_semantic_action
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


class MarketCanvasEnv(gym.Env):
    """Gymnasium environment for the MarketCanvas 2D design task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_steps: int = 50,
        canvas_width: int = 800,
        canvas_height: int = 600,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.canvas = Canvas(canvas_width, canvas_height)
        self.parsed_prompt = ParsedPrompt(raw="")
        self.step_count = 0

        # Semantic action space: dict with action_type and params
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(_ACTION_TYPES)),
            "x": spaces.Box(0, canvas_width, shape=(), dtype=np.int32),
            "y": spaces.Box(0, canvas_height, shape=(), dtype=np.int32),
            "width": spaces.Box(1, canvas_width, shape=(), dtype=np.int32),
            "height": spaces.Box(1, canvas_height, shape=(), dtype=np.int32),
            "element_type": spaces.Discrete(len(ElementType)),
            "element_idx": spaces.Discrete(21),  # max 20 elements + 0
            "z_index": spaces.Discrete(21),
        })

        # Observation: semantic state dict (JSON-serializable via info)
        self.observation_space = spaces.Dict({
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

        prompt = (options or {}).get("prompt", "Create a marketing banner")
        self.parsed_prompt = parse_prompt(prompt)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self._execute_action(action)

        reward_bd = compute_reward(self.canvas, self.parsed_prompt, self.step_count, self.max_steps)
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["reward_breakdown"] = reward_bd.to_dict()

        return obs, reward_bd.total, terminated, truncated, info

    def step_semantic(self, action_type: str, **params: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Convenience method for semantic actions with named parameters."""
        self.step_count += 1
        self._execute_semantic(action_type, params)

        reward_bd = compute_reward(self.canvas, self.parsed_prompt, self.step_count, self.max_steps)
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["reward_breakdown"] = reward_bd.to_dict()

        return obs, reward_bd.total, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return render_to_array(self.canvas)
        return None

    def get_state(self) -> dict[str, Any]:
        return self.canvas.to_dict()

    def get_reward(self) -> RewardBreakdown:
        return compute_reward(self.canvas, self.parsed_prompt, self.step_count, self.max_steps)

    def _get_obs(self) -> dict[str, Any]:
        return {"visual": render_to_array(self.canvas)}

    def _get_info(self) -> dict[str, Any]:
        return {
            "state": self.canvas.to_dict(),
            "step": self.step_count,
            "prompt": self.parsed_prompt.raw,
        }

    def _execute_action(self, action: dict[str, Any]) -> None:
        if "action_type" in action and isinstance(action["action_type"], str):
            self._execute_semantic(action["action_type"], action)
            return

        idx = int(action.get("action_type", 7))
        action_type = _ACTION_TYPES[idx] if idx < len(_ACTION_TYPES) else "noop"
        self._execute_semantic(action_type, action)

    def _execute_semantic(self, action_type: str, params: dict[str, Any]) -> None:
        apply_semantic_action(self.canvas, action_type, params)
