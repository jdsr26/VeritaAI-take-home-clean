"""Tests for MarketCanvas environment, actions, and reward."""

from __future__ import annotations

import json

import numpy as np
import pytest

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType
from marketcanvas.environment import ActionMode, MarketCanvasEnv


# ---------------------------------------------------------------------------
# 1. Reset returns correct observation keys and shapes
# ---------------------------------------------------------------------------

class TestResetObservation:
    def test_obs_keys_match_observation_space(self):
        env = MarketCanvasEnv()
        obs, info = env.reset(options={"prompt": "Create a banner"})

        # obs keys must exactly match the declared observation_space keys
        assert set(obs.keys()) == set(env.observation_space.spaces.keys())

    def test_obs_state_json_is_valid_json_with_elements(self):
        env = MarketCanvasEnv()
        obs, _ = env.reset(options={"prompt": "Create a banner"})

        state = json.loads(obs["state_json"])
        assert "canvas" in state
        assert "elements" in state
        assert isinstance(state["elements"], list)

    def test_obs_visual_shape(self):
        env = MarketCanvasEnv(canvas_width=400, canvas_height=300)
        obs, _ = env.reset()

        assert obs["visual"].shape == (300, 400, 3)
        assert obs["visual"].dtype == np.uint8


# ---------------------------------------------------------------------------
# 2. Semantic step adds an element and returns valid tuple
# ---------------------------------------------------------------------------

class TestSemanticStep:
    def test_add_element_increases_count(self):
        env = MarketCanvasEnv()
        env.reset(options={"prompt": "banner"})

        obs, reward, terminated, truncated, info = env.step_semantic(
            "add_element",
            type="text", content="Hello",
            x=10, y=10, width=200, height=50,
            color="#000000", text_color="#FFFFFF",
        )

        assert env.canvas.element_count() == 1
        state = json.loads(obs["state_json"])
        assert len(state["elements"]) == 1
        assert state["elements"][0]["content"] == "Hello"

    def test_move_element(self):
        env = MarketCanvasEnv()
        env.reset()
        env.step_semantic(
            "add_element", type="shape", content="",
            x=0, y=0, width=100, height=50,
        )
        el_id = env.canvas.elements[0].id

        env.step_semantic("move_element", id=el_id, new_x=200, new_y=300)

        moved = env.canvas.get_element(el_id)
        assert moved is not None
        assert moved.x == 200
        assert moved.y == 300

    def test_noop_does_not_change_canvas(self):
        env = MarketCanvasEnv()
        env.reset()
        state_before = env.canvas.to_dict()

        env.step_semantic("noop")

        assert env.canvas.to_dict() == state_before


# ---------------------------------------------------------------------------
# 3. Reward is in [-1, 1] and terminal_reward_only works
# ---------------------------------------------------------------------------

class TestReward:
    def test_reward_in_range(self):
        env = MarketCanvasEnv()
        env.reset(options={"prompt": "banner with headline"})

        # A few actions, then check reward bounds
        for _ in range(3):
            _, r, _, _, _ = env.step_semantic(
                "add_element", type="text", content="Hi",
                x=50, y=50, width=200, height=60,
                color="#111111", text_color="#FFFFFF",
            )
            assert -1.0 <= r <= 1.0, f"reward {r} out of [-1, 1]"

    def test_terminal_reward_only_returns_zero_on_intermediate_steps(self):
        env = MarketCanvasEnv(max_steps=3, terminal_reward_only=True)
        env.reset(options={"prompt": "banner"})

        _, r1, _, _, _ = env.step_semantic(
            "add_element", type="shape", content="",
            x=0, y=0, width=100, height=100,
        )
        assert r1 == 0.0, "intermediate step should return 0.0"

        _, r2, _, _, _ = env.step_semantic("noop")
        assert r2 == 0.0

        # Third step is terminal (truncated at max_steps=3)
        _, r3, _, truncated, info = env.step_semantic("noop")
        assert truncated is True
        assert info["is_terminal"] is True
        assert r3 != 0.0, "terminal step should return the actual reward"


# ---------------------------------------------------------------------------
# 4. Low-level action mode: drag creates shape, click selects, type appends
# ---------------------------------------------------------------------------

class TestLowLevelActions:
    def test_drag_on_empty_creates_shape(self):
        env = MarketCanvasEnv(action_mode="low_level")
        env.reset(options={"prompt": "banner"})

        env.step({"action_type": "mouse_drag", "x1": 100, "y1": 100, "x2": 400, "y2": 300})

        assert env.canvas.element_count() == 1
        el = env.canvas.elements[0]
        assert el.type == ElementType.SHAPE
        assert el.width == 300
        assert el.height == 200

    def test_click_selects_then_type_appends(self):
        env = MarketCanvasEnv(action_mode="low_level")
        env.reset()

        # Create a shape via drag
        env.step({"action_type": "mouse_drag", "x1": 50, "y1": 50, "x2": 250, "y2": 150})
        el_id = env.canvas.elements[0].id

        # Click on the created shape
        env.step({"action_type": "mouse_click", "x": 100, "y": 100})
        assert env.cursor_state.selected_element_id == el_id

        # Type text into it
        env.step({"action_type": "keyboard_type", "text": "Sale"})
        assert env.canvas.get_element(el_id).content == "Sale"

        env.step({"action_type": "keyboard_type", "text": " Now"})
        assert env.canvas.get_element(el_id).content == "Sale Now"


# ---------------------------------------------------------------------------
# 5. Episode truncation at max_steps
# ---------------------------------------------------------------------------

class TestEpisodeTruncation:
    def test_truncated_at_max_steps(self):
        env = MarketCanvasEnv(max_steps=2)
        env.reset()

        _, _, terminated1, truncated1, _ = env.step_semantic("noop")
        assert terminated1 is False
        assert truncated1 is False

        _, _, terminated2, truncated2, _ = env.step_semantic("noop")
        assert terminated2 is False
        assert truncated2 is True

    def test_reset_clears_step_count(self):
        env = MarketCanvasEnv(max_steps=2)
        env.reset()
        env.step_semantic("noop")
        env.step_semantic("noop")

        env.reset()
        assert env.step_count == 0
        assert env.canvas.element_count() == 0


# ---------------------------------------------------------------------------
# 6. Deterministic replay — same seed + same actions = same state
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_deterministic_replay(self):
        states = []
        for _ in range(2):
            env = MarketCanvasEnv()
            env.reset(seed=42, options={"prompt": "banner"})
            env.step_semantic("add_element", type="text", content="Hi",
                              x=10, y=10, width=200, height=50)
            env.step_semantic("add_element", type="shape", content="Buy",
                              x=100, y=200, width=150, height=60,
                              color="#FFFF00", text_color="#000000")
            states.append(env.canvas.to_dict())
        assert states[0] == states[1]
