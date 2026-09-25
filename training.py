"""Generate the eight saved learning stages without opening the application."""

import argparse

from src.model.checkpoints import DEFAULT_OUTPUT, train_checkpoints
from src.model.learning import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train_checkpoints(args.output, Config(seed=args.seed))


if __name__ == "__main__":
    main()
