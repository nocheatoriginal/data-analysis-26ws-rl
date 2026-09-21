import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from adventure.environment import Action, AdventureEnv
from adventure.learning import Config, QLearner, TrainingSession


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        check_env(AdventureEnv(), skip_render_check=True)

    def test_rewards_and_blocked_movement(self):
        env = AdventureEnv(("S.TG", ".#~."))
        self.assertEqual(env.reset(seed=1)[0], 0)
        self.assertEqual(env.step(Action.UP)[:4], (0, -5, False, False))
        self.assertEqual(env.step(Action.RIGHT)[:4], (1, -1, False, False))
        self.assertEqual(env.step(Action.DOWN)[:4], (1, -5, False, False))
        self.assertEqual(env.step(Action.RIGHT)[:4], (2, -15, False, False))
        self.assertEqual(env.step(Action.DOWN)[:4], (2, -5, False, False))
        self.assertEqual(env.step(Action.RIGHT)[:4], (3, 100, True, False))
        with self.assertRaises(RuntimeError):
            env.step(Action.LEFT)
        self.assertEqual(env.reset()[0], 0)

    def test_time_limit_is_truncation_and_goal_takes_priority(self):
        env = AdventureEnv(("SG",), max_steps=1)
        env.reset()
        self.assertEqual(env.step(Action.UP)[2:4], (False, True))
        env.reset()
        self.assertEqual(env.step(Action.RIGHT)[2:4], (True, False))

    def test_invalid_maps_and_actions(self):
        for layout in [(), ("S", "..G"), ("S#G",), ("SGX",), ("SSG",)]:
            with self.assertRaises(ValueError):
                AdventureEnv(layout)
        env = AdventureEnv()
        with self.assertRaises(ValueError):
            env.step(4)

    def test_ansi_render_and_state_encoding(self):
        env = AdventureEnv(("S.G",), render_mode="ansi")
        self.assertEqual(env.render(), "@ . G")
        self.assertEqual(env.position(2), (0, 2))


class LearningTests(unittest.TestCase):
    def test_numeric_q_update_and_terminal_bootstrap(self):
        agent = QLearner(3, Config(alpha=0.2, gamma=0.9))
        agent.q[0, 1] = 2
        agent.q[1, 2] = 10
        result = agent.update(0, 1, -1, 1, terminated=False)
        self.assertAlmostEqual(result["target"], 8)
        self.assertAlmostEqual(agent.q[0, 1], 3.2)
        agent.q[2] = 999  # Terminal Q-values must not leak into the target.
        result = agent.update(1, 1, 100, 2, terminated=True)
        self.assertEqual(result["target"], 100)
        self.assertEqual(agent.q[1, 1], 20)
        self.assertEqual(agent.visits[1, 1], 1)

    def test_truncation_keeps_bootstrap_in_training(self):
        session = TrainingSession(
            AdventureEnv(("S.G",), max_steps=1), Config(epsilon_start=0, epsilon_min=0)
        )
        session.agent.q[0, Action.RIGHT] = 1
        session.agent.q[1, Action.RIGHT] = 10
        update = session.step()
        self.assertTrue(update["truncated"])
        self.assertEqual(update["future"], 10)
        self.assertFalse(session.history[0]["success"])

    def test_reproducibility_and_epsilon_floor(self):
        config = Config(epsilon_decay=0.5, epsilon_min=0.1)
        a, b = TrainingSession(config=config), TrainingSession(config=config)
        a.train(20)
        b.train(20)
        np.testing.assert_array_equal(a.agent.q, b.agent.q)
        self.assertEqual(a.history, b.history)
        self.assertEqual(a.epsilon, 0.1)

    def test_evaluation_is_read_only_and_detects_loops(self):
        session = TrainingSession()
        session.train(10)
        q_before = session.agent.q.copy()
        visits_before = session.agent.visits.copy()
        state_before = (
            session.state,
            session.env.state,
            session.env.steps,
            session.total_steps,
            len(session.history),
        )
        rng_before = json.dumps(session.agent.rng.bit_generator.state)
        session.greedy_rollout()
        np.testing.assert_array_equal(session.agent.q, q_before)
        np.testing.assert_array_equal(session.agent.visits, visits_before)
        self.assertEqual(
            state_before,
            (
                session.state,
                session.env.state,
                session.env.steps,
                session.total_steps,
                len(session.history),
            ),
        )
        self.assertEqual(rng_before, json.dumps(session.agent.rng.bit_generator.state))
        self.assertEqual(TrainingSession().greedy_rollout()["reason"], "loop")

    def test_training_learns_optimal_discounted_return(self):
        # Value iteration provides an independent optimum for this exact reward
        # function, including routes through traps. It is never used by training.
        for seed in (7, 21, 42):
            with self.subTest(seed=seed):
                session = TrainingSession(config=Config(seed=seed))
                session.train(2000)
                result = session.greedy_rollout()
                self.assertTrue(result["success"])
                self.assertFalse(any(session.env.tile(s) == "T" for s in result["path"]))
                self.assertEqual(len(result["path"]), len(session.env.shortest_safe_path()))
                env, gamma = session.env, session.config.gamma
                values = np.zeros(env.observation_space.n)
                for _ in range(500):
                    updated = values.copy()
                    for state in range(len(values)):
                        if env.tile(state) in "#~G":
                            continue
                        candidates = []
                        for action in Action:
                            nxt, reward, terminal, _ = env.transition(state, action)
                            candidates.append(reward + (0 if terminal else gamma * values[nxt]))
                        updated[state] = max(candidates)
                    if np.max(np.abs(updated - values)) < 1e-10:
                        values = updated
                        break
                    values = updated
                discounted = 0
                for t, state in enumerate(result["path"][:-1]):
                    _, reward, _, _ = env.transition(state, int(np.argmax(session.agent.q[state])))
                    discounted += gamma**t * reward
                self.assertAlmostEqual(discounted, values[env.start], places=7)

    def test_exports_are_complete(self):
        session = TrainingSession()
        session.train(3)
        with tempfile.TemporaryDirectory() as directory:
            session.export(directory)
            with (Path(directory) / "q_table.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 80)
            self.assertIn("UP", rows[0])
            with (Path(directory) / "episodes.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 3)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
