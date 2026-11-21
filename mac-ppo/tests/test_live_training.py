"""Check that the live screen is attached to actual fresh optimization."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]


class LiveTrainingTest(unittest.TestCase):
    def test_live_run_starts_fresh_updates_weights_and_preserves_demo_model(self):
        frozen=ROOT/"models/pong_ppo.zip"
        digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/"fresh"
            env=os.environ.copy()
            env["SDL_VIDEODRIVER"]="dummy"
            result=subprocess.run([sys.executable,str(ROOT/"train.py"),"--live","--steps","1024",
                "--envs","1","--out",str(output),"--close-on-finish"],cwd=ROOT,env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            status=json.loads((output/"live-status.json").read_text())
            self.assertEqual(status["initialization"],"random weights")
            self.assertTrue(status["finished"])
            self.assertEqual(status["timesteps"],1024)
            self.assertGreater(status["adam_steps"],0)
            self.assertGreater(status["weight_change_l2"],0)
            first=PPO.load(output/"initial.zip",device="cpu")
            last=PPO.load(output/"latest.zip",device="cpu")
            self.assertEqual(first.num_timesteps,0)
            self.assertGreater(last.num_timesteps,0)
            self.assertTrue(any(not torch.equal(t,last.policy.state_dict()[k])
                                for k,t in first.policy.state_dict().items()))
            original=hashlib.sha256((output/"initial.zip").read_bytes()).hexdigest()
            repeat=subprocess.run([sys.executable,str(ROOT/"train.py"),"--steps","1024",
                "--envs","1","--out",str(output)],cwd=ROOT,env=env,
                capture_output=True,text=True,timeout=30)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual(hashlib.sha256((output/"initial.zip").read_bytes()).hexdigest(),original)
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),digest)


if __name__=="__main__":
    unittest.main()
