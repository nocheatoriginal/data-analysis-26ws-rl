"""Prepare missing snapshots, then open their read-only playback interface."""

import argparse
import os

from .model.checkpoints import CHECKPOINTS, DEFAULT_OUTPUT, ensure_checkpoints
from .model.learning import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=int, choices=CHECKPOINTS, default=1)
    parser.add_argument("--screenshot", metavar="PNG")
    parser.add_argument("--diagnose-display", action="store_true")
    args = parser.parse_args()
    if args.diagnose_display and args.screenshot:
        parser.error("--diagnose-display requires a real desktop window")
    config = Config(seed=args.seed)
    ensure_checkpoints(args.output, config)
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    from .ui.dashboard import Dashboard

    dashboard = Dashboard(args.output, config, args.checkpoint, args.diagnose_display)
    if args.screenshot:
        dashboard.screenshot(args.screenshot)
    else:
        dashboard.run()


if __name__ == "__main__":
    main()
