"""Real ALE Pong with compact, visible-object observations.

The extractor reads the current image only. It neither chooses an action nor
looks ahead in the emulator. All movement decisions come from the neural net.
"""

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

FEATURES = [
    "ball_x", "ball_y", "ball_vx", "ball_vy", "player_y", "opponent_y",
    "ball_relative_y", "ball_relative_x", "player_velocity", "ball_visible",
]
ACTIONS = (0, 2, 3)  # ALE NOOP, RIGHT (= paddle up), LEFT (= paddle down).
ACTION_NAMES = ("STAY", "UP", "DOWN")


class PongObjects(gym.Env):
    """An unmodified Pong ROM; only observations/actions/rewards are wrapped."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, *, shaping=False, sticky=0.25, max_steps=15000):
        self.env = gym.make(
            "ALE/Pong-v5", obs_type="rgb", render_mode="rgb_array",
            frameskip=4, repeat_action_probability=sticky, max_episode_steps=max_steps,
        )
        self.observation_space = gym.spaces.Box(-1, 1, (len(FEATURES),), np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.render_mode = "rgb_array"
        self.shaping = shaping
        self.last_ball = None
        self.last_player = 114.0
        self.player_y = 114.0
        self.opponent_y = 114.0
        self.last_features = np.zeros(len(FEATURES), dtype=np.float32)
        self.frame = None
        self.player_score = self.opponent_score = 0

    @staticmethod
    def _paddle_y(image, x0, x1, red, fallback):
        ys, _ = np.where(image[34:194, x0:x1, 0] == red)
        if not len(ys):
            return fallback
        lo, hi = float(ys.min() + 34), float(ys.max() + 34)
        if hi - lo < 15:
            return hi - 7.5 if lo == 34 else lo + 7.5
        return float(ys.mean() + 34)

    def _features(self, frame):
        player = self._paddle_y(frame, 138, 145, 92, self.player_y)
        opponent = self._paddle_y(frame, 14, 21, 213, self.opponent_y)
        ys, xs = np.where(frame[34:194, 20:140, 0] == 236)
        visible = bool(len(xs))
        bx, by, vx, vy = 80.0, 114.0, 0.0, 0.0
        if visible:
            bx, by = float(xs.mean() + 20), float(ys.mean() + 34)
            if self.last_ball is not None:
                dx, dy = bx - self.last_ball[0], by - self.last_ball[1]
                # Exclude a new serve's discontinuous jump.
                if abs(dx) <= 16 and abs(dy) <= 16:
                    vx, vy = dx / 8.0, dy / 8.0
            self.last_ball = bx, by
        else:
            self.last_ball = None
        result = np.array([
            (bx - 80) / 80, (by - 114) / 80, vx, vy,
            (player - 114) / 80, (opponent - 114) / 80,
            (by - player) / 160, (bx - 140) / 160,
            (player - self.last_player) / 12, float(visible),
        ], dtype=np.float32)
        self.last_player, self.player_y, self.opponent_y = player, player, opponent
        self.frame = frame
        self.last_features = np.clip(result, -1, 1)
        return self.last_features.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_ball = None
        self.last_player = self.player_y = self.opponent_y = 114.0
        self.player_score = self.opponent_score = 0
        frame, info = self.env.reset(seed=seed, options=options)
        return self._features(frame), info

    @staticmethod
    def _potential(features):
        # A small dense learning signal, based only on the current visible image.
        return -0.5 * abs(float(features[6])) * float(features[9])

    def step(self, action):
        old_features = self.last_features.copy()
        frame, raw_reward, terminated, truncated, info = self.env.step(ACTIONS[int(action)])
        if raw_reward > 0:
            self.player_score += int(raw_reward)
        elif raw_reward < 0:
            self.opponent_score -= int(raw_reward)
        features = self._features(frame)
        reward = float(raw_reward)
        hit = bool(old_features[2] > 0 and features[2] < 0 and old_features[0] > 0.4)
        if self.shaping:
            reward *= 5.0
            # Auxiliary reward is used for training only, never for reported scores.
            reward += 0.1 * float(hit)
            next_potential = 0.0 if terminated else self._potential(features)
            reward += 0.995 * next_potential - self._potential(old_features)
        info.update({
            "raw_reward": float(raw_reward), "player_score": self.player_score,
            "opponent_score": self.opponent_score, "paddle_hit": hit,
        })
        return features, reward, terminated, truncated, info

    def render(self):
        return self.frame.copy()

    def close(self):
        self.env.close()


def make_env(*, shaping=False, sticky=0.25):
    return PongObjects(shaping=shaping, sticky=sticky)
