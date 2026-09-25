# relearn — Q-learning simulation

A Python project for exploring reinforcement learning in a small adventure game.
An agent learns to reach a treasure by exploring an 8 × 10 grid, avoiding traps,
walls and rivers. The Pygame dashboard visualizes the world, the complete Q-table,
individual updates, and the learning curve. No pretrained policy or image assets
are needed.

## Run

Requires Python 3.10 or newer. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src
```

On Windows, activate with `.venv\Scripts\activate` instead. `python main.py` is an
equivalent entry point. The UI uses a light background, restrained colors and
simple controls. The board stays visible while the right-hand tabs switch between
**Learning curve**, **Selected tile**, and **Learning step**. Clicking a tile opens
its values; **Step** opens the latest calculation. Keyboard shortcuts remain
available even though they are no longer repeated on every button.

The window can be resized (minimum 1080 × 720). Text is rendered with FreeType at
the actual output pixel density, using the font's natural weight. On a 2× Retina
display, a 14-point label is rasterized at 28 pixels rather than enlarging a small
bitmap. Some Pygame 2.6 builds silently map `allow_highdpi` to zero; `display.py`
repairs that flag before creating the window. Shapes and text rendering remain
separate from the learning algorithm.

Start with **Step** to inspect individual decisions, then **Train +500** and
**Watch**. Watch plays the greedy policy without exploration or Q-table changes.
A policy that has not learned enough may loop; playback detects and reports this.

| Control | What it does |
| --- | --- |
| Run / Space | Start or pause continuous training |
| Step / N | Pause and learn exactly one transition; select the updated state |
| Train +500 | Train 500 more episodes at accelerated speed, then pause |
| Watch / W | Play a greedy evaluation from the start, with a frozen Q-table |
| View / Q | Cycle through policy arrows, four Q-values per tile, visits, and full table |
| Click a tile or table row | Inspect that state's four action values and visit counts |
| Mouse wheel in table view | Scroll through all 80 states |
| Speed / keys 1–4 | Choose 4, 30, 300 or 3,000 training transitions per second |
| Export / E | Write Q-values, episode data and evaluation to `outputs/` |
| Reset / R | Clear learning and restart using the same seed and settings |
| Escape | Close the application |

Requested training speed depends on the computer. Accelerated training stays
responsive by limiting work per frame. Playback always uses a slower animation.
Policy arrows resolve ties in action order; training breaks ties randomly.
Unvisited states show a dot. In the Q-grid, the four values are positioned up,
right, down and left and rounded to integers; the inspector shows two decimals.
The small blue dot indicates the agent in every grid view. Blocked tiles and the
terminal goal have no action decisions; their unused table rows stay zero.

## Train and export without a window

```sh
python -m src --headless --episodes 2000 --seed 7
python -m src --episodes 2000
python -m src --headless --episodes 2000 --screenshot outputs/dashboard.png
```

The second command opens an already trained dashboard. A screenshot uses an
offscreen display and works without a desktop session.

Exports are `q_table.csv` (all states, actions and visit counts), `episodes.csv`
(return, length, success, exploration rate), and `summary.json` (configuration,
map, rewards and greedy evaluation). Exporting replaces these files in the chosen
directory; use `--output outputs/experiment-2` to keep experiments separate.
The exports are analysis artifacts, not resumable training checkpoints.

For seed 7, 2,000 episodes with the defaults learn a trap-free route of **17 moves**,
with an undiscounted return of **+84**: 16 ordinary moves at −1, then +100 at the
goal. An independent breadth-first search finds a shortest trap-free route of
17 moves. That reference search never supplies actions or values to the learner.
Tests also compare the learned route's discounted return with value iteration.
Finite training is not a general guarantee of optimality on arbitrary maps or
parameter settings.

## Game rules

The state is the agent's tile: `state = row * columns + column`, with zero-based
coordinates. There are 80 states and four actions, ordered **up, right, down,
left**, so the Q-table has shape `(80, 4)`.

| Tile or event | Reward | Effect |
| --- | ---: | --- |
| `S` start / `.` free path | −1 | Move to the tile |
| `T` trap | −15 | Move to the tile; the episode continues |
| `#` wall / `~` river / outside the grid | −5 | Stay in the current tile |
| `G` treasure / goal | +100 | End the episode successfully |
| 250-step limit | Ordinary action reward | Truncate the episode and restart |

Rewards are alternatives, not additive: a goal move earns +100, not +99.
Every re-entry to a trap incurs its penalty. Dead ends cost extra moves, and
attempting to leave through a blocked edge incurs the blocked-move penalty.
Rivers cannot be crossed; this introductory project uses only four movement actions.

## How the agent learns

1. Start at `S` with a Q-table initialized to zero.
2. With probability epsilon, choose a random action (**explore**). Otherwise,
   choose an action with the highest current Q-value (**exploit**).
3. Observe the next state and reward; update only the chosen state-action entry.
4. Repeat until the goal or time limit, then start another episode and reduce epsilon.

```text
target = reward + gamma * max Q(next_state, action)
Q(state, action) += alpha * (target - Q(state, action))
```

At the goal, `target = reward`: there is no future return. At an external time
limit the bootstrap term remains, since the next state is not terminal.
For example, with old Q = 2, reward = −1, next best Q = 10, alpha = 0.2 and
gamma = 0.9, the target is 8 and the new Q-value is 3.2.

**Alpha** controls how quickly new observations change an estimate. **Gamma**
discounts distant rewards. **Epsilon** controls exploration and decays from 1.0
by a factor of 0.997 per episode to a floor of 0.05. The agent learns expected
discounted return; the chart displays undiscounted episode return, averaged over
the most recent 25 episodes at each point. The success statistic uses the last
100 completed training episodes, including their exploratory actions.

Try changing one setting at a time:

```sh
python -m src --alpha 0.1 --gamma 0.98 --epsilon-decay 0.998 --seed 21
python -m src --headless --episodes 2000 --epsilon 0.1 --output outputs/low-exploration
```

Compare learning speed, episode returns, success rates and the final route. Very
low exploration can leave promising paths undiscovered. A low discount factor
can make immediate costs dominate distant treasure. Edit `DEFAULT_MAP` in
`src/model/environment.py` to experiment with level design. The grid layout
adapts to the number of rows and columns, but keep maps small so the four Q-values
per cell stay readable. The environment validates tile symbols and checks that
the goal is reachable.

## Gymnasium and neural-network context

`AdventureEnv` implements the modern [Gymnasium environment interface](https://gymnasium.farama.org/api/env/),
the maintained successor to OpenAI Gym. It exposes discrete observation/action
spaces and separates goal termination from time-limit truncation:

```python
from src.model.environment import AdventureEnv

env = AdventureEnv(render_mode="ansi")
observation, info = env.reset(seed=7)
for _ in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

The [Gymnasium custom-environment guide](https://gymnasium.farama.org/introduction/create_custom_env/)
explains this interface. The dashboard is a separate
[Pygame](https://www.pygame.org/docs/) renderer; `env.render()` provides a text grid.

In this project, the input is a tile index and the output is four stored action
values. In a driving task, inputs might instead be camera images, speed and sensor
distances, while actions could represent steering and acceleration choices. A
neural network can approximate Q-values when a table becomes too large, as in
deep Q-learning. The reward/transition/learning loop still applies, but training a
network introduces additional machinery such as replay memory and a target
network. This project deliberately implements tabular Q-learning only; it does
not include a driving simulator, neural network, or the referenced Udemy course.

## Structure and verification

```text
main.py                   Start with python main.py
src/
    __main__.py           Command-line options and application startup
    model/
        environment.py    Grid rules and Gymnasium interface
        learning.py       Q-table, training, evaluation and export
    ui/
        dashboard.py      Drawing, input handling and interactive training loop
        display.py        HiDPI window and text rendering
        theme.py          Colors, dimensions and labels
        diagnostics.py    Optional desktop display diagnostics
tests/                    Environment, learning and UI checks
```

`model/` has no dependency on the UI or Pygame. `ui/` reads the model and calls
its training methods in response to input. `__main__.py` parses command-line
options and starts either headless training or the dashboard. This is a simple
UI/logic split; drawing and input handling stay together in the dashboard rather
than introducing a separate MVC controller. Each package also has a small
`__init__.py` file so Python can import it normally.

Run commands from the project root. `python main.py` and `python -m src` use the
same entry point; the previous `python -m adventure` command is replaced.


Generated files are not part of the source code. `outputs/` is created only when
you export results, save screenshots there, or enable display diagnostics. Its
contents can be removed when no longer needed. Python and development-tool caches
(`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`) can also be removed; they are
regenerated automatically. These directories are ignored by Git. `.venv/` contains
the installed dependencies and should stay in place while working on the project.

For reading or changing the code, start with `environment.py` (what an action
does), then `QLearner.update()` (the equation) and `TrainingSession.step()` (one
learning step). The UI calls that same step method whether training is manual
or automatic. Learning parameters have one source of defaults in `Config`.

The UI deliberately uses one dashboard class and ordinary drawing functions.
`update_layout()` calculates rectangles shared by drawing and mouse clicks;
`draw_*()` methods render each section; `handle_event()` and `action()` handle
input; `advance()` schedules training and playback. There is no UI framework,
custom widget hierarchy or separate event system. Visual constants live in
`theme.py`, independently of the game rules. The light palette and plain system
fonts keep the focus on the simulation.

`display.py` isolates the Pygame SDL2 window/renderer API used for HiDPI support.
Layout and clicks use window coordinates; text rendering uses physical pixels.
This distinction matters on Retina displays, where an ordinary Pygame software
window can otherwise be enlarged by macOS. See the
[Pygame HiDPI issue](https://github.com/pygame/pygame/issues/2853).
Restart the application after updating the code so it creates the new window.

If the live window still looks blurry, record its actual output configuration:

```sh
python -m src --diagnose-display
```

While that window is open, `outputs/display-diagnostics.json` is refreshed every
five seconds. It records the interpreter, SDL renderer, HiDPI flag, window size,
physical renderer output, viewport, and frame size. The output size is queried
directly from SDL, independently of the viewport. This command intentionally
cannot be combined with `--screenshot` or `--headless`: an offscreen image cannot
verify how the desktop compositor displays the real window. Diagnostics are
opt-in and do not change system display settings.

```sh
python -m unittest discover -s tests -v
```

Optional development tools provide consistent formatting and basic lint checks:

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
```

Use `python -m ruff format .` to format changes. Settings live in `pyproject.toml`;
Ruff is not needed to run the simulation.

Checks cover Gymnasium compatibility, movement/rewards, terminal and truncated
episodes, exact Q-update arithmetic, repeatable training, three training seeds,
independent route optimality, read-only evaluation, exports, dashboard rendering,
selection, playback, training controls, and resizing without changing font sizes.
A separate test checks that 2× text matches freshly rasterized, double-resolution
glyphs, rather than an enlarged 1× image. The automated tests simulate this pixel
density; the visual result on a physical Retina monitor still depends on its
display settings and must be checked on that monitor.
Tested with Python 3.12, NumPy 1.26.4,
Pygame 2.6.1 and Gymnasium 1.3.0.
