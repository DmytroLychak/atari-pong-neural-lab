"""Observe the real training rollout and optimizer, without choosing actions."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
import torch
from stable_baselines3 import PPO


class ObservablePPO(PPO):
    """Exactly PPO's update, with a UI notification immediately before/after."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_live_window"]

    def train(self):
        window = getattr(self, "_live_window", None)
        if window:
            window.optimizer_started()
        super().train()
        if window:
            window.optimizer_finished()


class TrainingWindow:
    def __init__(self, model, vec, output, target_steps, *, fresh=True, fps=16, screenshots=None):
        self.model, self.vec, self.output = model, vec, Path(output)
        self.target_steps, self.fresh, self.fps = target_steps, fresh, fps
        self.screenshots = Path(screenshots) if screenshots else None
        if self.screenshots:
            self.screenshots.mkdir(parents=True, exist_ok=True)
        self.initial = torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach().clone()
        self.weight_change = 0.0
        self.adam_steps = 0
        self.start = time.perf_counter()
        self.last_draw = self.last_status = 0.0
        self.steps = model.num_timesteps
        self.episodes = []
        self.player = self.opponent = 0
        self.last_game = "Очікуємо завершення першої навчальної гри"
        self.state = "ЗБІР ДОСВІДУ"
        self.paused = self.slow = self.stop_requested = self.closed = self.finished = False
        self.frame = vec.env_method("render", indices=0)[0]
        self.action = 0
        self.probabilities = np.ones(3) / 3
        self.milestones = [100000, 700000, 1500000]
        self.captured = set()
        pygame.init()
        desktop=pygame.display.get_desktop_sizes()[0]
        scale=min(1,(desktop[0]-60)/1180,(desktop[1]-100)/830)
        self.display=pygame.display.set_mode((int(1180*scale),int(830*scale)))
        self.screen=pygame.Surface((1180,830))
        pygame.display.set_caption("Pong — живе навчання PPO з випадкових ваг")
        self.font = pygame.font.SysFont("Arial", 22)
        self.small = pygame.font.SysFont("Arial", 18)
        self.title = pygame.font.SysFont("Arial", 31, bold=True)
        self.clock = pygame.time.Clock()
        self.colors = dict(bg=(16, 23, 36), text=(235, 241, 248), muted=(163, 180, 199),
                           green=(97, 224, 160), yellow=(255, 206, 98))
        self.draw()
        self.capture("initial.png")
        self.write_status()

    def text(self, label, x, y, *, color=None, font=None):
        self.screen.blit((font or self.font).render(label, True, color or self.colors["text"]), (x, y))

    def capture(self, filename):
        if self.screenshots:
            pygame.image.save(self.screen.convert(24), self.screenshots / filename)

    def write_status(self):
        status = dict(timesteps=self.steps, adam_steps=self.adam_steps, weight_change_l2=self.weight_change,
                      episodes=len(self.episodes), elapsed_seconds=time.perf_counter()-self.start,
                      initialization="random weights" if self.fresh else "explicit resume", finished=self.finished,
                      state=self.state, paused=self.paused, preview="actual training environment 1",
                      last_50_mean_raw_return=float(np.mean([r[1] for r in self.episodes[-50:]])) if self.episodes else None)
        temp=self.output/"live-status.tmp"
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(temp,self.output/"live-status.json")

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                self.stop_requested=self.closed=True
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_SPACE and not self.finished:
                    self.paused=not self.paused
                elif event.key==pygame.K_n and not self.finished:
                    self.slow=not self.slow
                elif event.key==pygame.K_s:
                    self.stop_requested=True
                elif event.key==pygame.K_RETURN and self.finished:
                    subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1]/"play.py"),
                                      "--model",str(self.output/"latest.zip")])
                    self.closed=True

    def observe(self, local_variables, steps):
        self.steps=steps
        infos=local_variables.get("infos",[])
        if infos:
            first=infos[0]
            self.player,self.opponent=first.get("player_score",0),first.get("opponent_score",0)
            if "episode" in first:
                self.player=self.opponent=0
            for info in infos:
                if "episode" in info:
                    raw=info["player_score"]-info["opponent_score"]
                    self.episodes.append((steps,raw))
                    self.last_game=f"Остання завершена гра: {info['player_score']}:{info['opponent_score']}"
        now=time.perf_counter()
        if now-self.last_draw < 1/self.fps and not self.slow:
            return not self.stop_requested
        self.events()
        while self.paused and not self.stop_requested:
            self.draw()
            self.events()
            self.clock.tick(20)
        if self.closed:
            return False
        self.frame=self.vec.env_method("render",indices=0)[0]
        if "actions" in local_variables:
            self.action=int(local_variables["actions"][0])
        if "obs_tensor" in local_variables:
            with torch.no_grad():
                self.probabilities=self.model.policy.get_distribution(local_variables["obs_tensor"][:1]).distribution.probs[0].cpu().numpy()
        self.draw()
        self.last_draw=now
        if now-self.last_status>=1:
            self.write_status();self.last_status=now
        for milestone in self.milestones:
            if steps>=milestone and milestone not in self.captured:
                self.capture(f"step_{milestone:08d}.png");self.captured.add(milestone)
        if self.slow:
            self.clock.tick(15)
        return not self.stop_requested

    def optimizer_started(self):
        if not self.closed:
            self.state="ОНОВЛЕННЯ ВАГ PPO / ADAM"
            self.events()
            while self.paused and not self.stop_requested:
                self.draw();self.events();self.clock.tick(20)
            self.frame=self.vec.env_method("render",indices=0)[0]
            self.draw()

    def optimizer_finished(self):
        vector=torch.nn.utils.parameters_to_vector(self.model.policy.parameters()).detach()
        self.weight_change=float(torch.linalg.vector_norm(vector-self.initial))
        first=next(iter(self.model.policy.optimizer.state.values()),{})
        self.adam_steps=int(first.get("step",0))
        self.state="ЗБІР ДОСВІДУ"
        if not self.closed:
            self.draw()
            if self.adam_steps>0 and "first_update" not in self.captured:
                self.capture("first_weight_update.png");self.captured.add("first_update")

    def draw(self):
        self.screen.fill(self.colors["bg"])
        title="Pong · живе навчання з нуля" if self.fresh else "Pong · продовження навчання"
        self.text(title,26,17,font=self.title)
        self.text("PPO збирає новий досвід і змінює ваги під час цього запуску",26,58,font=self.small,color=self.colors["muted"])
        surface=pygame.surfarray.make_surface(np.transpose(self.frame,(1,0,2)))
        self.screen.blit(pygame.transform.scale(surface,(480,630)),(26,102))
        self.text(f"Навчальне середовище 1 із {self.vec.num_envs}",26,747,font=self.small)
        self.text("Показуються реальні кадри тренування з пропуском",26,775,font=self.small,color=self.colors["muted"])
        x=540
        self.text("ПАУЗА" if self.paused else self.state,x,104,color=self.colors["yellow"])
        self.text(f"Переходи: {self.steps:,} / {self.target_steps:,}".replace(","," "),x,145)
        elapsed=max(time.perf_counter()-self.start,.001)
        self.text(f"Кроків Adam: {self.adam_steps:,}   Час: {elapsed:.0f} с".replace(","," "),x,180)
        self.text(f"Зміна ваг від початку: {self.weight_change:.4f}",x,215,color=self.colors["green"])
        self.text(f"Поточний рахунок: {self.player}:{self.opponent}",x,250)
        self.text(self.last_game,x,284,font=self.small,color=self.colors["muted"])
        self.text(f"Навчальних ігор у 6 середовищах: {len(self.episodes)}" if self.vec.num_envs==6
                  else f"Завершених навчальних ігор: {len(self.episodes)}",x,316,font=self.small)
        labels=["Стояти","Угору","Униз"]
        for i,label in enumerate(labels):
            y=355+i*29
            self.text(label,x,y,font=self.small,color=self.colors["green"] if self.action==i else None)
            pygame.draw.rect(self.screen,(40,54,73),(x+115,y+3,260,14))
            pygame.draw.rect(self.screen,self.colors["green"],(x+115,y+3,int(260*self.probabilities[i]),14))
            self.text(f"{self.probabilities[i]:.1%}",x+390,y,font=self.small)
        self.text("Різниця очок: середнє останніх 50 навчальних ігор",x,459,font=self.small)
        chart=pygame.Rect(x+40,498,505,168)
        for value in [-20,0,20]:
            yy=chart.bottom-int((value+21)/42*chart.height)
            pygame.draw.line(self.screen,(55,69,87),(chart.left,yy),(chart.right,yy))
            self.text(str(value),x,yy-10,font=self.small,color=self.colors["muted"])
        if len(self.episodes)>1:
            values=np.array([r[1] for r in self.episodes],dtype=float)
            means=[values[max(0,i-49):i+1].mean() for i in range(len(values))]
            points=[(chart.left+int(r[0]/self.target_steps*chart.width),
                     chart.bottom-int((mean+21)/42*chart.height)) for r,mean in zip(self.episodes,means)]
            pygame.draw.lines(self.screen,self.colors["green"],False,points,2)
        else:self.text("Ще немає завершених ігор",x+80,560,font=self.small,color=self.colors["muted"])
        self.text("0",chart.left,chart.bottom+5,font=self.small,color=self.colors["muted"])
        self.text(f"{self.target_steps/1e6:g} млн переходів",chart.right-150,chart.bottom+5,font=self.small,color=self.colors["muted"])
        if self.finished:
            self.text("Enter — зіграти новими щойно навченими вагами",x,714,font=self.small,color=self.colors["green"])
            self.text("Esc — закрити. Ваги збережено у папці запуску.",x,748,font=self.small)
        else:
            self.text("Space — пауза     S — зупинити й зберегти",x,714,font=self.small)
            self.text("N — наочно / швидко     Esc — зберегти й закрити",x,748,font=self.small)
            self.text("Темп: наочно, 15 кроків/с" if self.slow else "Темп: швидке навчання; кадри з пропуском",x,785,font=self.small,color=self.colors["yellow"])
        self.display.blit(pygame.transform.smoothscale(self.screen,self.display.get_size()),(0,0))
        pygame.display.flip()

    def finish(self, *, stopped=False, close=False):
        self.finished=True
        self.paused=False
        self.state="ЗУПИНЕНО · ВАГИ ЗБЕРЕЖЕНО" if stopped else "НАВЧАННЯ ЗАВЕРШЕНО"
        if not self.closed:
            self.draw();self.capture("finished.png")
        self.write_status()
        while not self.closed and not close:
            self.events();self.draw();self.clock.tick(15)
        pygame.quit()
