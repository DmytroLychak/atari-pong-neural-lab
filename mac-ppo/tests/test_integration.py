"""Small integration checks against the real emulator and submitted weights."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from stable_baselines3 import PPO

from pong_ai.environment import make_env
from pong_ai.checkpoint import save_model


class IntegrationTests(unittest.TestCase):
    def test_same_seed_and_actions_reproduce_real_environment(self):
        a, b = make_env(), make_env()
        try:
            oa, _ = a.reset(seed=411)
            ob, _ = b.reset(seed=411)
            np.testing.assert_array_equal(oa, ob)
            for action in [0, 1, 2, 1] * 30:
                oa, ra, ta, xa, ia = a.step(action)
                ob, rb, tb, xb, ib = b.step(action)
                np.testing.assert_array_equal(oa, ob)
                self.assertEqual((ra, ta, xa), (rb, tb, xb))
                self.assertTrue(a.observation_space.contains(oa))
        finally:
            a.close()
            b.close()

    def test_submitted_model_uses_valid_actions_and_original_scores(self):
        torch.set_num_threads(1)
        model = PPO.load(Path(__file__).resolve().parents[1] / "models/pong_ppo.zip", device="cpu")
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/"roundtrip.zip"
            save_model(model,path)
            restored=PPO.load(path,device="cpu")
            for name, tensor in model.policy.state_dict().items():
                self.assertTrue(torch.equal(tensor,restored.policy.state_dict()[name]),name)
        env = make_env()
        total = 0.0
        try:
            obs, _ = env.reset(seed=811)
            for _ in range(1500):
                action = int(model.predict(obs, deterministic=True)[0])
                self.assertTrue(env.action_space.contains(action))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                self.assertEqual(total, info["player_score"] - info["opponent_score"])
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
