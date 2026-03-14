"""Demo script showcasing MarketCanvas-Env with three agent baselines."""

from __future__ import annotations

import argparse
import json
import random

from marketcanvas.elements import ElementType
from marketcanvas.environment import MarketCanvasEnv
from marketcanvas.renderer import save_png


DEFAULT_PROMPT = "Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast"


def print_reward(label: str, env: MarketCanvasEnv) -> None:
    bd = env.get_reward()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Constraint:    {bd.constraint:.3f}")
    print(f"  Layout:        {bd.layout:.3f}")
    print(f"  Accessibility: {bd.accessibility:.3f}")
    print(f"  Completeness:  {bd.completeness:.3f}")
    print(f"  TOTAL REWARD:  {bd.total:.3f}")
    print(f"  Elements:      {env.canvas.element_count()}")
    print(f"  Steps taken:   {env.step_count}")


def print_final_state(label: str, env: MarketCanvasEnv) -> None:
    print(f"\n  Final semantic state ({label}):")
    print(json.dumps(env.get_state(), indent=2, sort_keys=True))


def run_nop_baseline(prompt: str) -> None:
    """Zero actions on a blank canvas — should score near -1.0."""
    print("\n" + "#"*60)
    print("  BASELINE 1: NOP (No Actions)")
    print("#"*60)

    env = MarketCanvasEnv()
    env.reset(options={"prompt": prompt})
    print_reward("NOP Baseline", env)
    print_final_state("NOP", env)
    save_png(env.canvas, "output_nop.png")
    print("  Saved: output_nop.png")


def run_oracle(prompt: str) -> None:
    """Scripted perfect design — should score near +1.0."""
    print("\n" + "#"*60)
    print("  BASELINE 2: Oracle (Scripted Perfect Design)")
    print("#"*60)

    env = MarketCanvasEnv()
    env.reset(options={"prompt": prompt})

    # Step 1: Add a background banner shape
    _, r, _, _, info = env.step_semantic(
        "add_element",
        type="shape", content="",
        x=0, y=0, width=800, height=600,
        color="#1A1A5E", text_color="#FFFFFF",
    )
    print(f"  Step 1 (banner bg):   reward={r:.3f}")

    # Step 2: Add headline text
    _, r, _, _, _ = env.step_semantic(
        "add_element",
        type="text", content="Summer Sale — Up to 50% Off!",
        x=100, y=80, width=600, height=80,
        color="#1A1A5E", text_color="#FFFFFF",
    )
    print(f"  Step 2 (headline):    reward={r:.3f}")

    # Step 3: Add subtitle text
    _, r, _, _, _ = env.step_semantic(
        "add_element",
        type="text", content="Limited time offer on all items",
        x=150, y=200, width=500, height=50,
        color="#1A1A5E", text_color="#CCCCCC",
    )
    print(f"  Step 3 (subtitle):    reward={r:.3f}")

    # Step 4: Add yellow CTA button
    _, r, _, _, _ = env.step_semantic(
        "add_element",
        type="shape", content="Shop Now",
        x=275, y=350, width=250, height=70,
        color="#FFFF00", text_color="#000000",
    )
    print(f"  Step 4 (CTA button):  reward={r:.3f}")

    # Step 5: Add decorative image placeholder
    _, r, _, _, _ = env.step_semantic(
        "add_element",
        type="image", content="product image",
        x=550, y=420, width=200, height=150,
        color="#2A2A7E",
    )
    print(f"  Step 5 (image):       reward={r:.3f}")

    print_reward("Oracle Result", env)
    print_final_state("Oracle", env)
    save_png(env.canvas, "output_oracle.png")
    print("  Saved: output_oracle.png")


def run_random_agent(prompt: str, n_steps: int = 15) -> None:
    """Random actions — should score between -0.5 and 0.0."""
    print("\n" + "#"*60)
    print(f"  BASELINE 3: Random Agent ({n_steps} steps)")
    print("#"*60)

    env = MarketCanvasEnv()
    env.reset(options={"prompt": prompt})

    random.seed(42)
    elem_types = list(ElementType)
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#000000", "#FFFFFF"]

    for step in range(n_steps):
        action_type = random.choice(["add_element", "move_element", "change_color", "noop", "add_element"])

        if action_type == "add_element":
            _, r, _, _, _ = env.step_semantic(
                "add_element",
                type=random.choice(elem_types).value,
                content=random.choice(["text", "click me", "hello", "sale", ""]),
                x=random.randint(0, 700),
                y=random.randint(0, 500),
                width=random.randint(30, 300),
                height=random.randint(30, 200),
                color=random.choice(colors),
                text_color=random.choice(colors),
            )
        elif action_type == "move_element" and env.canvas.element_count() > 0:
            el = random.choice(env.canvas.elements)
            _, r, _, _, _ = env.step_semantic(
                "move_element", id=el.id,
                new_x=random.randint(0, 700),
                new_y=random.randint(0, 500),
            )
        elif action_type == "change_color" and env.canvas.element_count() > 0:
            el = random.choice(env.canvas.elements)
            _, r, _, _, _ = env.step_semantic(
                "change_color", id=el.id, hex_code=random.choice(colors),
            )
        else:
            _, r, _, _, _ = env.step_semantic("noop")

        print(f"  Step {step+1:2d} ({action_type:15s}): reward={r:.3f}")

    print_reward("Random Agent Result", env)
    print_final_state("Random", env)
    save_png(env.canvas, "output_random.png")
    print("  Saved: output_random.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketCanvas demo baselines")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Target design prompt used to score all baselines",
    )
    parser.add_argument(
        "--random-steps",
        type=int,
        default=15,
        help="Number of steps for the random baseline",
    )
    args = parser.parse_args()

    print("MarketCanvas-Env Demo")
    print(f"Prompt: {args.prompt}\n")

    run_nop_baseline(args.prompt)
    run_oracle(args.prompt)
    run_random_agent(args.prompt, n_steps=args.random_steps)

    print("\n" + "="*60)
    print("  Demo complete! Check output_*.png files for visual results.")
    print("="*60)


if __name__ == "__main__":
    main()
