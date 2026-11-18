"""Atari Pong: object features, imitation learning and independent evaluation.

The RGB feature extractor includes an analytic bounce prediction. The neural
network learns motor control from demonstrations; this is NOT end-to-end RL.
No teacher is called during neural-policy evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent
ACTIONS = (0, 2, 3)  # NOOP, RIGHT (up in Pong), LEFT (down in Pong)
CONFIG = dict(env_id="ALE/Pong-v5", frameskip=1, repeat_action_probability=0.25,
              full_action_space=False, max_episode_steps=108000)


def make_env():
    cfg = CONFIG.copy()
    return gym.make(cfg.pop("env_id"), **cfg)


class Features:
    """Seven normalized features, using RGB only (no RAM or emulator state).

    x_ball, y_ball, vx_ball, vy_ball, y_paddle, visibility, predicted target.
    Reflection assumes ball center boundaries y=35.5 and y=192.5. The target
    feature is an explicit hand-engineered physics prior, not a learned CNN.
    """
    def __init__(self):
        self.previous = None

    def __call__(self, rgb):
        yy, xx = np.where(rgb[34:194, :, 0] == 236)
        py = np.where(rgb[34:194, 140:144, 1] == 186)[0]
        paddle = float(py.mean() + 34) if len(py) else 114.0
        bx, by, dx, dy, target = 80.0, 114.0, 0.0, 0.0, 114.0
        visible = bool(len(yy))
        if visible:
            bx, by = float(xx.mean()), float(yy.mean() + 34)
            if self.previous is not None:
                dx, dy = bx - self.previous[0], by - self.previous[1]
                target = by
                if 0 < dx < 10 and abs(dy) < 10:
                    raw = by + dy * (139 - bx) / dx
                    target = 35.5 + (raw - 35.5) % 314
                    if target > 192.5:
                        target = 385 - target
            self.previous = (bx, by)
        else:
            self.previous = None
        return np.array([bx / 160, (by - 34) / 160,
                         np.clip(dx / 8, -1, 1), np.clip(dy / 8, -1, 1),
                         (paddle - 34) / 160, float(visible),
                         (target - 34) / 160], dtype=np.float32)


def teacher(features):
    """Demonstrator labels are used exclusively for data collection."""
    delta = (float(features[6]) - float(features[4])) * 160
    return 1 if delta < -2 else 2 if delta > 2 else 0


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.layers(x)


def neural_action(model, features):
    with torch.inference_mode():
        return int(model(torch.from_numpy(features)).argmax().item())


def episode(seed, mode, model=None, collect=False, beta=0.0, video=None):
    env = make_env()
    obs, _ = env.reset(seed=int(seed))
    extractor = Features()
    rng = np.random.default_rng(seed + 98765)
    x, y = [], []
    player = opponent = 0
    started = time.perf_counter()
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(video), fps=30, macro_block_size=2)
    try:
        for step in range(CONFIG["max_episode_steps"]):
            features = extractor(obs)
            # This branch is never entered by evaluate/play.
            if collect:
                expert = teacher(features)
                x.append(features)
                y.append(expert)
                action = expert if mode == "teacher" or rng.random() < beta else neural_action(model, features)
            elif mode == "random":
                action = int(rng.integers(3))
            elif mode == "teacher":
                action = teacher(features)
            else:
                action = neural_action(model, features)
            if writer and step % 2 == 0:
                writer.append_data(obs)
            obs, reward, terminated, truncated, _ = env.step(ACTIONS[action])
            player += int(reward > 0)
            opponent += int(reward < 0)
            if terminated or truncated:
                break
        if writer:
            writer.append_data(obs)
            from PIL import Image
            Image.fromarray(obs).resize((480, 630), Image.Resampling.NEAREST).save(Path(video).with_suffix(".png"))
    finally:
        env.close()
        if writer:
            writer.close()
    row = dict(seed=int(seed), policy=mode, player_score=player, opponent_score=opponent,
               return_=player-opponent, win=bool(terminated and player == 21),
               terminated=bool(terminated), truncated=bool(truncated), steps=step+1,
               seconds=round(time.perf_counter()-started, 3))
    return row, x, y


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def summary(rows):
    values = np.array([r["return_"] for r in rows])
    n, wins = len(rows), sum(r["win"] for r in rows)
    p, z = wins / n, 1.96
    c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(episodes=n, wins=wins, win_rate=p, win_rate_wilson95=[max(0,c-h),min(1,c+h)],
                mean_return=float(values.mean()), std_return=float(values.std(ddof=1)) if n>1 else 0,
                min_return=int(values.min()), max_return=int(values.max()),
                truncated=sum(r["truncated"] for r in rows))


def evaluate(model, seeds, mode="neural"):
    if model is not None:
        model.eval()
    rows=[]
    for seed in seeds:
        row, _, _ = episode(seed, mode, model)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def evaluate_worker(job):
    torch.set_num_threads(1)
    seed, mode, model_path = job
    model = load_model(model_path) if mode == 'neural' else None
    row, _, _ = episode(seed, mode, model)
    return row


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    model = Policy()
    results, models = ROOT / "results", ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models / "initial.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    all_x, all_y, episodes, history, validation = [], [], [], [], []
    started = time.perf_counter()
    # Eight training games: four teacher games, then two DAgger rounds of two.
    for round_idx, seeds in enumerate((range(100,104), range(200,202), range(300,302))):
        for seed in seeds:
            row, x, y = episode(seed, "teacher" if round_idx==0 else "neural", model,
                                collect=True, beta=0.2 if round_idx==1 else 0.0)
            row["round"] = round_idx
            episodes.append(row)
            all_x.extend(x)
            all_y.extend(y)
            print("COLLECT", json.dumps(row), "dataset",len(all_y),flush=True)
        xx, yy = torch.tensor(np.array(all_x)), torch.tensor(all_y, dtype=torch.long)
        model.train()
        for epoch in range(15):
            perm=torch.randperm(len(yy))
            loss_sum=correct=0
            for idx in perm.split(1024):
                logits=model(xx[idx])
                loss=nn.functional.cross_entropy(logits, yy[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*len(idx)
                correct += int((logits.argmax(1)==yy[idx]).sum())
            record=dict(round=round_idx, epoch=epoch+1, samples=len(yy),
                        loss=loss_sum/len(yy), accuracy=correct/len(yy))
            history.append(record)
            print("EPOCH",json.dumps(record),flush=True)
        torch.save(model.state_dict(), models/f"round_{round_idx}.pt")
        rows=evaluate(model, range(1000,1003))
        validation.append(dict(round=round_idx, summary=summary(rows), episodes=rows))
        save_json(results/"training_progress.json",dict(episodes=episodes,history=history,validation=validation))
    # Checkpoint choice uses validation only, before the unseen final test.
    selected = max(validation, key=lambda v: (v['summary']['wins'], v['summary']['mean_return']))['round']
    shutil.copyfile(models/f"round_{selected}.pt", models/"policy.pt")
    np.savez_compressed(results/"demonstrations.npz", x=np.array(all_x), y=np.array(all_y))
    metadata=dict(config=CONFIG, algorithm="behavioral cloning + DAgger",
                  seed=42, training_seeds=[100,101,102,103,200,201,300,301],
                  validation_seeds=[1000,1001,1002], epochs_per_round=15,
                  batch_size=1024, learning_rate=0.001, optimizer="Adam",
                  architecture=[7,64,64,3], parameters=sum(p.numel() for p in model.parameters()),
                  samples=len(all_y), seconds=time.perf_counter()-started,
                  python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                  gymnasium=gym.__version__,ale_py=ale_py.__version__,device="cpu",
                  selected_round=selected,
                  policy_sha256=hashlib.sha256((models/"policy.pt").read_bytes()).hexdigest())
    save_json(results/"training_metadata.json", metadata)
    print("TRAINING COMPLETE",json.dumps(metadata),flush=True)


def load_model(path):
    model=Policy()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    model.eval()
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("train")
    ev=sub.add_parser("evaluate")
    ev.add_argument("--episodes",type=int,default=20)
    ev.add_argument("--workers",type=int,default=1)
    ev.add_argument("--resume",action="store_true",help="Resume the same protocol from saved episode rows")
    ev.add_argument("--seed",type=int,default=10000)
    ev.add_argument("--model",type=Path,default=ROOT/"models/policy.pt")
    ev.add_argument("--mode",choices=["neural","random"],default="neural")
    ev.add_argument("--output",type=Path,default=ROOT/"results/evaluation.json")
    play=sub.add_parser("play")
    play.add_argument("--seed",type=int,default=10000)
    play.add_argument("--model",type=Path,default=ROOT/"models/policy.pt")
    play.add_argument("--video",type=Path,default=ROOT/"results/demo.mp4")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.command=="train":
        train(args)
    elif args.command=="evaluate":
        if args.episodes<1:parser.error("--episodes must be positive")
        model_hash=hashlib.sha256(args.model.read_bytes()).hexdigest() if args.mode=='neural' else None
        rows=[]
        if args.resume and args.output.exists():
            old=json.loads(args.output.read_text(encoding='utf-8'))
            if old['config']!=CONFIG or old.get('policy_sha256')!=model_hash:
                raise ValueError('Resume requires identical environment and model')
            rows=old['episodes']
            valid_seeds=set(range(args.seed,args.seed+args.episodes))
            if any(r['policy']!=args.mode or r['seed'] not in valid_seeds for r in rows):
                raise ValueError('Resume policy or seed range mismatch')
        completed={r['seed'] for r in rows}
        jobs=[(seed,args.mode,args.model) for seed in range(args.seed,args.seed+args.episodes) if seed not in completed]
        def persist():
            rows.sort(key=lambda r:r['seed'])
            report=dict(config=CONFIG,summary=summary(rows),episodes=rows)
            if model_hash is not None:report['policy_sha256']=model_hash
            save_json(args.output,report)
            return report
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for row in pool.map(evaluate_worker,jobs):
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                    persist()
        else:
            for job in jobs:
                row=evaluate_worker(job)
                rows.append(row)
                print(json.dumps(row),flush=True)
                persist()
        report=persist()
        with args.output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
        print("SUMMARY",json.dumps(report["summary"]),flush=True)
    else:
        args.video.parent.mkdir(parents=True,exist_ok=True)
        row,_,_=episode(args.seed,"neural",load_model(args.model),video=args.video)
        save_json(args.video.with_suffix(".json"),row)
        print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
