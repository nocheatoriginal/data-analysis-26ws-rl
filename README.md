# relearn — Q-learning simulation

A small reinforcement-learning teaching project. An agent learns to reach a goal
in an 8 × 10 grid. Training runs **before the application opens**. The interface
plays saved episodes and learned routes without changing the Q-table.

## Start

Requires Python 3.10 or newer. From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, use `.venv\Scripts\activate`. `python -m src` is an equivalent entry
point. On first startup, the application prepares all eight checkpoints in
`output/`. Subsequent starts reuse them. Missing, unreadable or incompatible
checkpoints trigger a new training run before the window opens.

To generate or regenerate the files explicitly without opening the UI:

```sh
python training.py
```

Both commands accept `--output DIRECTORY` and `--seed NUMBER`. Use the same seed
and directory for training and playback. Defaults are seed 7 and the project's
`output/` folder, independent of the working directory. A different seed, map,
reward configuration or checkpoint format invalidates the old files.

```sh
python training.py --seed 21 --output output/seed-21
python main.py --seed 21 --output output/seed-21 --checkpoint 500
python main.py --checkpoint 5000 --screenshot output/dashboard.png
```

The previous live-training CLI options (`--headless`, `--episodes`, etc.) are
replaced by `training.py`. Learning defaults are defined in `Config` in
`src/model/learning.py`.

## Saved stages and playback

One continuous training run saves its state after exactly **1, 50, 100, 200, 500,
1,000, 3,000 and 5,000 completed episodes**:

```text
output/
    q_table_1.npy
    q_table_1.json
    q_table_50.npy
    q_table_50.json
    ...
    q_table_5000.npy
    q_table_5000.json
```

Each `.npy` file contains the `(80, 4)` Q-table. The accompanying JSON contains
map/settings, cumulative visit counts, learning history and the actual steps of
that checkpoint's training episode. Both files are needed for the application.
A Q-table alone cannot reconstruct the random actions taken during training.
These snapshots are for inspection and playback, not for resuming training.

Choose a checkpoint using the numbered buttons. The mode button switches between:

- **Training episode:** replay the actual recorded episode, including exploration,
  walls, traps and failed moves. A timed-out episode plays all of its recorded
  steps. The step panel shows the reward, exploration decision and original
  Q-value update. The displayed table and policy arrows show the saved values at
  the **end** of the selected episode, not a reconstruction of every earlier table.
- **Learned route:** start at `START` and follow the highest Q-value in the saved
  table without exploration. Repeated states stop playback with a loop result,
  after showing the move that caused it. Ties use action order. This evaluation
  may differ from the actual training episode.

A red arrow marks an invalid move; the agent stays in place. Water and traps are
traversable and their penalties appear in the step panel. Playback never updates
Q-values or visit counts. Loaded arrays are read-only.

| Control | Action |
| --- | --- |
| Number buttons / keys 1–8 | Select a saved stage |
| Play / Space | Play or pause |
| Restart / R | Rewind the selected playback |
| Mode button / W | Switch recorded episode / learned route |
| View / Q | Policy, Q-values, visits, full table |
| Speed button | 1, 5, 10 or 20 moves per second |
| Click tile or table row | Inspect its action values |
| Mouse wheel in table | Scroll all states |
| Escape | Close |

The learning curve shows episode return averaged over up to 25 episodes. The
success statistic uses the last 100 completed training episodes. All metrics
belong to the selected checkpoint. The right-hand tabs separate the learning
curve, selected tile and playback step to keep the interface simple.

## Game rules

| Tile or action | Reward | Effect |
| --- | ---: | --- |
| START / EMPTY | −1 | Move to the tile |
| WATER | −3 | Move into and through water |
| WALL / outside the grid | −5 | Stay in place |
| TRAP | −20 | Move to the tile; episode continues |
| GOAL | +100 | End successfully |
| 250-step limit | Ordinary action reward | End the episode as truncated |

Rewards are alternatives, not additive. Every entry into a tile incurs its
reward; resetting to START does not itself count as a move.

Maps use `Tile` enum members. Edit `DEFAULT_MAP` in `src/model/environment.py`:

```python
from src.model.environment import AdventureEnv, Tile

layout = (
    (Tile.START, Tile.WATER, Tile.GOAL),
    (Tile.EMPTY, Tile.WALL, Tile.TRAP),
)
env = AdventureEnv(layout)
```

The map must be rectangular, with exactly one start and one reachable goal.
Symbols are only display labels. State indices are `row * columns + column`.
Actions are ordered UP, RIGHT, DOWN, LEFT.

## Learning

Training uses epsilon-greedy action selection: with probability epsilon, choose a
random action; otherwise choose a highest-valued action (breaking ties randomly).
After each transition, update only its state/action entry:

```text
target = reward + gamma * max Q(next_state)
Q(state, action) += alpha * (target - Q(state, action))
```

At the goal, future return is zero. At the external step limit, the next state's
value still contributes. Alpha controls learning speed, gamma discounts future
rewards, and epsilon decays after each completed episode.

With seed 7, the default 5,000-episode checkpoint learns a 15-move route with
return +84, crossing water once. Early checkpoints can loop or take worse routes.
The tests compare learned discounted return with independent value iteration;
a shortest-path search alone cannot assess different water and trap costs.
Neither reference algorithm supplies actions to the learner.

## Code structure

```text
main.py                       Application entry point
training.py                   Generate saved stages without a window
src/
    __main__.py               Prepare snapshots and start the application
    model/
        environment.py        Tile enum, map, movement and rewards
        learning.py           Q-learning and training sessions
        checkpoints.py        Save/load stages and recorded episodes
    ui/
        dashboard.py          Read-only playback, drawing and controls
        display.py            HiDPI window and text rendering
        theme.py              Colors and layout constants
        diagnostics.py        Optional desktop display diagnostics
tests/                        Environment, learning, storage and UI checks
```

The model does not depend on Pygame. `TrainingSession.step()` performs one learning
transition; `train_checkpoints()` saves stages at episode boundaries. The UI calls
`load_checkpoint()` and advances a recording with `playback_step()`. It never
calls the training step. `update_layout()` shares rectangles between rendering and
click detection. `draw_*()` methods render the individual sections.

`output/` contains generated data and is ignored by Git. It can be deleted to
force preparation on the next start. Python/Ruff caches are also regenerable.
Keep `.venv/`, which contains the project's installed dependencies.

## Display and verification

The light UI can be resized (minimum 1080 × 720). FreeType renders text at the
actual output pixel density. `display.py` fixes the HiDPI flag on affected Pygame
2.6 builds. On a 2× Retina display, a 14-point label uses 28-pixel glyphs.

To inspect a real window's configuration:

```sh
python main.py --diagnose-display
```

This refreshes `output/display-diagnostics.json` every five seconds. It cannot
be combined with `--screenshot`, which uses an offscreen renderer.

```sh
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m ruff check .
```

Tests cover movement (including crossing water), rewards, Gymnasium compatibility,
Q-update arithmetic, optimal discounted return, exact checkpoint boundaries,
recorded actions, checkpoint reuse, immutable playback, UI controls and HiDPI text.
