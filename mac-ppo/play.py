"""A live, local demo. The displayed actions are selected by the trained net."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from pong_ai.environment import ACTION_NAMES, make_env

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/pong_ppo.zip")
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--speed", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--capture-at", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. See README for setup/training.")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    model.policy.set_training_mode(False)
    env = make_env()
    observation, _ = env.reset(seed=args.seed)
    pygame.init()
    screen = pygame.display.set_mode((1100, 780))
    pygame.display.set_caption("Pong AI — Личак Дмитро — нейронна мережа PPO")
    font = pygame.font.SysFont("Arial", 22)
    small = pygame.font.SysFont("Arial", 18)
    title = pygame.font.SysFont("Arial", 32, bold=True)
    clock = pygame.time.Clock()
    colors = dict(bg=(16, 23, 36), text=(235, 241, 248), muted=(163, 180, 199), accent=(97, 224, 160))
    pause = False
    running = True
    games = wins = total_steps = 0
    speed = args.speed
    seed = args.seed
    action = 0
    info = {"player_score": 0, "opponent_score": 0}
    probabilities = np.ones(3) / 3
    message = "Мережа керує зеленою ракеткою праворуч"

    def text(label, x, y, color=None, f=None):
        screen.blit((f or font).render(label, True, color or colors["text"]), (x, y))

    def reset_episode():
        nonlocal observation, seed, info
        seed += 1
        observation, _ = env.reset(seed=seed)
        info = {"player_score": 0, "opponent_score": 0}

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        pause = not pause
                    elif event.key == pygame.K_r:
                        reset_episode()
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_4):
                        speed = int(event.unicode)
            if not running:
                break
            if not pause:
                tensor = torch.as_tensor(observation[None], dtype=torch.float32)
                with torch.no_grad():
                    probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
                action = int(np.argmax(probabilities))
                observation, _, terminated, truncated, info = env.step(action)
                total_steps += 1
                if terminated or truncated:
                    games += 1
                    won = bool(terminated and info["player_score"] > info["opponent_score"])
                    wins += int(won)
                    reason = "перемога" if won else ("ліміт часу" if truncated else "поразка")
                    message = f"Партія {games}: {info['player_score']}:{info['opponent_score']} — {reason}"
                    print(json.dumps({"seed": seed, "player": info["player_score"], "opponent": info["opponent_score"],
                                      "terminated": bool(terminated), "truncated": bool(truncated)}, ensure_ascii=False), flush=True)
                    reset_episode()
            screen.fill(colors["bg"])
            text("Pong · навчена нейромережа", 28, 20, f=title)
            text("Справжня Atari 2600 / ALE · локально на Mac · ваги PPO", 28, 62, colors["muted"], small)
            frame = env.render()
            game_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.scale(game_surface, (480, 630)), (28, 102))
            x = 544
            text("РАХУНОК ПОТОЧНОЇ ПАРТІЇ", x, 109, colors["muted"], small)
            text(f"Мережа {info['player_score']} : {info['opponent_score']} Комп’ютер", x, 141, colors["accent"], title)
            text(f"Партій: {games}     Перемог: {wins}", x, 190)
            text("РІШЕННЯ МЕРЕЖІ", x, 244, colors["muted"], small)
            labels = ["Стояти", "Угору", "Униз"]
            for i, label in enumerate(labels):
                y = 280 + i * 54
                text(label, x, y, colors["accent"] if action == i else colors["text"])
                pygame.draw.rect(screen, (40, 54, 73), (x + 140, y + 3, 225, 20), border_radius=3)
                pygame.draw.rect(screen, colors["accent"], (x + 140, y + 3, int(225 * probabilities[i]), 20), border_radius=3)
                text(f"{probabilities[i]:.1%}", x + 382, y, f=small)
            text("10 ознак → 64 → 64 → 3 дії", x, 467)
            text("Actor обирає argmax імовірностей.", x, 502, colors["muted"], small)
            text("Під час показу ваги не змінюються.", x, 530, colors["muted"], small)
            text(f"Seed: {seed}   Швидкість: ×{speed}", x, 594, colors["muted"], small)
            text("Space — пауза    R — нова партія", x, 628, f=small)
            text("1 / 2 / 4 — швидкість    Esc — вихід", x, 657, f=small)
            text("ПАУЗА" if pause else "ГРА ТРИВАЄ", x, 702, colors["accent"])
            text(message, 28, 749, colors["muted"], small)
            pygame.display.flip()
            if args.screenshot and total_steps == args.capture_at:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, args.screenshot)
            if args.max_steps and total_steps >= args.max_steps:
                running = False
            if not args.headless:
                clock.tick(15 * speed)
    finally:
        env.close()
        pygame.quit()
    print("Demo closed cleanly.", flush=True)


if __name__ == "__main__":
    main()
