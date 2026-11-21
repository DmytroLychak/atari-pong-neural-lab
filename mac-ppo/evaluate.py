"""Evaluate frozen weights on fresh seeds using the original Pong score."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from pong_ai.environment import make_env


def evaluate(model_path, *, episodes, seed, sticky=0.25):
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu") if model_path else None
    env = make_env(shaping=False, sticky=sticky)
    rows = []
    start = time.perf_counter()
    for number in range(episodes):
        episode_seed = seed + number
        observation, _ = env.reset(seed=episode_seed)
        rng = np.random.default_rng(episode_seed)
        score = 0.0
        for step in range(15000):
            action = int(model.predict(observation, deterministic=True)[0]) if model else int(rng.integers(3))
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                row = dict(seed=episode_seed, return_=score, player=info["player_score"],
                           opponent=info["opponent_score"], steps=step + 1,
                           terminated=bool(terminated), truncated=bool(truncated),
                           won=bool(terminated and info["player_score"] > info["opponent_score"]))
                rows.append(row)
                print(json.dumps(row), flush=True)
                break
    env.close()
    returns = np.array([r["return_"] for r in rows])
    n = len(rows)
    wins = sum(r["won"] for r in rows)
    # Wilson's 95% interval for a binomial win-rate estimate.
    z = 1.959963984540054
    phat = wins / n
    center = (phat + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {
        "model": str(model_path) if model_path else "uniform random action baseline",
        "environment": "ALE/Pong-v5", "frameskip": 4, "sticky_action_probability": sticky,
        "shaped_rewards": False, "deterministic_policy": model is not None,
        "episodes": n, "first_seed": seed, "wins": wins, "win_rate": phat,
        "win_rate_wilson_95": [center-half, center+half],
        "mean_return": float(returns.mean()), "std_return": float(returns.std(ddof=1)) if n > 1 else 0.0,
        "min_return": float(returns.min()), "max_return": float(returns.max()),
        "mean_player_points": float(np.mean([r["player"] for r in rows])),
        "mean_opponent_points": float(np.mean([r["opponent"] for r in rows])),
        "truncated_episodes": sum(r["truncated"] for r in rows),
        "elapsed_seconds": time.perf_counter()-start, "games": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--sticky", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    data = evaluate(args.model, episodes=args.episodes, seed=args.seed, sticky=args.sticky)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k:v for k,v in data.items() if k != "games"}), flush=True)


if __name__ == "__main__":
    main()
