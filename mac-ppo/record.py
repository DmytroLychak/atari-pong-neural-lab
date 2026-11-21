"""Record one complete, uncut game as an offline backup for the live demo."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
from stable_baselines3 import PPO

from pong_ai.environment import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/pong_ppo.zip"))
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--out", type=Path, default=Path("results/demo.mp4"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    env = make_env()
    obs, _ = env.reset(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Each transition advances four Atari frames. 15 fps is real time; 30 is x2.
    with imageio.get_writer(args.out, fps=30, codec="libx264", macro_block_size=2, quality=8) as writer:
        writer.append_data(env.render())
        for step in range(15000):
            action = int(model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            writer.append_data(env.render())
            if terminated or truncated:
                data = {"seed": args.seed, "model": str(args.model), "transitions": step+1,
                        "player": info["player_score"], "opponent": info["opponent_score"],
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "playback_speed": 2, "video_fps": 30, "cut_or_edited": False}
                args.out.with_suffix(".json").write_text(json.dumps(data, indent=2))
                print(json.dumps(data), flush=True)
                break
    env.close()


if __name__ == "__main__":
    main()
