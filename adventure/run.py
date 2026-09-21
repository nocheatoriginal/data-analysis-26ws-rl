"""Run with python -m adventure; train without a window using --headless."""

import argparse
import os

from .environment import AdventureEnv
from .learning import Config, TrainingSession


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Interactive Q-learning grid adventure")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Train, evaluate and export without opening a window",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes to train before opening the UI (headless default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--alpha", type=float, default=defaults.alpha, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="Discount factor")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults.epsilon_start,
        help="Initial exploration probability",
    )
    parser.add_argument("--epsilon-min", type=float, default=defaults.epsilon_min)
    parser.add_argument("--epsilon-decay", type=float, default=defaults.epsilon_decay)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument(
        "--output", default="outputs", help="Export directory (existing export files are replaced)"
    )
    parser.add_argument(
        "--screenshot", metavar="PNG", help="Render a dashboard image without opening a window"
    )
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 0:
        parser.error("--episodes must be nonnegative")
    try:
        config = Config(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        )
        session = TrainingSession(AdventureEnv(max_steps=args.max_steps), config)
    except ValueError as exc:
        parser.error(str(exc))
    episodes = args.episodes
    if episodes is None:
        episodes = 2000 if args.headless else 0
    if episodes:
        session.train(episodes)
    if args.headless:
        path = session.export(args.output)
        result = session.greedy_rollout()
        reference = session.env.shortest_safe_path()
        print(f"Trained {len(session.history)} episodes ({session.total_steps:,} transitions).")
        print(
            f"Greedy evaluation: {result['reason']}; {len(result['path']) - 1} moves; return {result['return']:+.0f}."
        )
        print(f"Shortest trap-free reference: {len(reference) - 1} moves.")
        print(f"Exports: {path.resolve()}")
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    if not args.headless or args.screenshot:
        from .ui import Dashboard

        dashboard = Dashboard(session, args.output)
        if args.screenshot:
            dashboard.screenshot(args.screenshot)
            print(f"Screenshot: {args.screenshot}")
        else:
            dashboard.run()


if __name__ == "__main__":
    main()
