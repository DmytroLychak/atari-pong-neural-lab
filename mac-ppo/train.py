"""Train PPO from random weights, with reproducible checkpoints and CSV logs."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pong_ai.environment import FEATURES, make_env


class SaveProgress(BaseCallback):
    def __init__(self, output, interval):
        super().__init__()
        self.output, self.interval = output, interval
        self.next_save = interval
        self.start = time.perf_counter()

    def _on_training_start(self):
        self.next_save = (self.num_timesteps // self.interval + 1) * self.interval
        path = self.output / "episodes.csv"
        self.episodes_file = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.episodes_file, fieldnames=[
            "timesteps", "player", "opponent", "raw_return", "length", "shaped_return"])
        if path.stat().st_size == 0:
            self.writer.writeheader()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.writer.writerow(dict(timesteps=self.num_timesteps,
                    player=info["player_score"], opponent=info["opponent_score"],
                    raw_return=info["player_score"]-info["opponent_score"],
                    length=info["episode"]["l"], shaped_return=info["episode"]["r"]))
        if self.num_timesteps >= self.next_save:
            self.episodes_file.flush()
            self.model.save(self.output / "latest")
            self.model.save(self.output / f"step_{self.num_timesteps:08d}")
            data = {"timesteps": self.num_timesteps, "elapsed_seconds": time.perf_counter() - self.start}
            (self.output / "progress.json").write_text(json.dumps(data, indent=2))
            print("CHECKPOINT", json.dumps(data), flush=True)
            self.next_save += self.interval
        return not (self.output / "STOP").exists()

    def _on_training_end(self):
        self.episodes_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs/pong_ppo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.check:
        env = make_env(shaping=True)
        check_env(env)
        env.close()
    constructors = [lambda: Monitor(make_env(shaping=True)) for _ in range(args.envs)]
    vec = SubprocVecEnv(constructors, start_method="spawn") if args.envs > 1 else DummyVecEnv(constructors)
    vec.seed(args.seed)
    options = dict(
        learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=6,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh},
        device="cpu", seed=args.seed, verbose=1,
    )
    config = {k: str(v) if k == "policy_kwargs" else v for k, v in options.items()}
    config.update({"features": FEATURES, "num_envs": args.envs, "requested_additional_steps": args.steps,
                   "environment": "ALE/Pong-v5", "frameskip": 4, "sticky_action_probability": 0.25,
                   "initialization": str(args.resume) if args.resume else "random weights"})
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    model = PPO.load(args.resume, env=vec, device="cpu") if args.resume else PPO("MlpPolicy", vec, **options)
    if not args.resume:
        model.save(args.out / "initial")
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(args.out), ["csv"]))
    try:
        model.learn(args.steps, callback=SaveProgress(args.out, 100_000), reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(args.out / "latest")
        vec.close()
    print("Saved", args.out / "latest.zip", flush=True)


if __name__ == "__main__":
    main()
