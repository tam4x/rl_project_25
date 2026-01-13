import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy


# ---------- Helpers ----------
def safe_name(s: str) -> str:
    """Make task names safe for filenames."""
    return (
        s.replace(" ", "_")
         .replace("=", "")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
    )


# ---------- Wrappers ----------
class TargetVelocity(Wrapper):
    """
    Generic target-velocity reward wrapper (works for HalfCheetah and Walker2d).
    Reward = -|vx - target_v| * vel_scale - ctrl_cost_weight * ||a||^2
    """
    def __init__(self, env, target_velocity: float, vel_scale: float = 1.0, ctrl_cost_weight: float = 0.1):
        super().__init__(env)
        self.vt = float(target_velocity)
        self.vel_scale = float(vel_scale)
        self.ctrl_cost_weight = float(ctrl_cost_weight)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # In MuJoCo locomotion envs: qvel[0] is usually forward x-velocity
        vx = float(self.env.unwrapped.data.qvel[0])
        vel_reward = -abs(vx - self.vt) * self.vel_scale
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        reward = vel_reward - ctrl_cost

        info = dict(info)
        info.update({"vx": vx, "target_v": self.vt, "vel_reward": vel_reward, "ctrl_cost": ctrl_cost})
        return obs, reward, terminated, truncated, info


class Walker2dJump(Wrapper):
    """
    Optional: Jump task for Walker2d (harder).
    Reward = beta * max(0, height - h0) - ctrl_cost_weight * ||a||^2
    """
    def __init__(self, env, baseline_height: float = 1.25, beta: float = 5.0, ctrl_cost_weight: float = 0.001):
        super().__init__(env)
        self.h0 = float(baseline_height)
        self.beta = float(beta)
        self.ctrl_cost_weight = float(ctrl_cost_weight)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Walker2d torso height often accessible via qpos[1]
        h = float(self.env.unwrapped.data.qpos[1])
        jump_reward = self.beta * max(0.0, h - self.h0)
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        reward = jump_reward - ctrl_cost

        info = dict(info)
        info.update({"height": h, "h0": self.h0, "jump_reward": jump_reward, "ctrl_cost": ctrl_cost})
        return obs, reward, terminated, truncated, info


# ---------- Task factories ----------
def task_base(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def task_halfcheetah_target_velocity(target_v: float, seed: int = 0,
                                     vel_scale: float = 1.0, ctrl_cost_weight: float = 0.1):
    env = gym.make("HalfCheetah-v4")
    env = TargetVelocity(env, target_velocity=target_v, vel_scale=vel_scale, ctrl_cost_weight=ctrl_cost_weight)
    env.reset(seed=seed)
    return env


def task_walker2d_target_velocity(target_v: float, seed: int = 0,
                                  vel_scale: float = 1.0, ctrl_cost_weight: float = 0.001):
    env = gym.make("Walker2d-v4")
    env = TargetVelocity(env, target_velocity=target_v, vel_scale=vel_scale, ctrl_cost_weight=ctrl_cost_weight)
    env.reset(seed=seed)
    return env


def task_walker2d_jump(seed: int = 0, baseline_height: float = 1.25, beta: float = 5.0, ctrl_cost_weight: float = 0.001):
    env = gym.make("Walker2d-v4")
    env = Walker2dJump(env, baseline_height=baseline_height, beta=beta, ctrl_cost_weight=ctrl_cost_weight)
    env.reset(seed=seed)
    return env


# ---------- Task dataclass ----------
@dataclass(frozen=True)
class Task:
    name: str
    make_env: Callable[[], gym.Env]


# ---------- Vec env builder ----------
def build_vec_env(task: Task, seed: int = 0, normalize_obs: bool = True):
    def _init():
        env = task.make_env()
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([_init])

    if normalize_obs:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return venv


# ---------- Teacher builder ----------
def make_teacher(algo: str, env, seed: int = 0, logdir: str = None):
    algo = algo.upper()
    common = dict(verbose=1, seed=seed, tensorboard_log=logdir)

    if algo == "SAC":
        return SAC("MlpPolicy", env, batch_size=256, learning_rate=3e-4, gamma=0.99, **common)

    if algo == "TD3":
        return TD3("MlpPolicy", env, batch_size=256, learning_rate=1e-3, gamma=0.99, **common)

    if algo == "PPO":
        return PPO("MlpPolicy", env, n_steps=2048, batch_size=64, learning_rate=3e-4, gamma=0.99, **common)

    raise ValueError(f"Unknown algo: {algo}")


# ---------- Train + save teacher ----------
def train_teacher_for_task(
    task: Task,
    algo: str = "SAC",
    total_timesteps: int = 300_000,
    seed: int = 0,
    normalize_obs: bool = True,
    out_dir: str = "./teachers",
    log_dir: str = "./tb_logs",
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    venv = build_vec_env(task, seed=seed, normalize_obs=normalize_obs)

    model = make_teacher(algo, venv, seed=seed, logdir=log_dir)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    venv.training = False
    venv.norm_reward = False

    mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=10, deterministic=True)
    print(f"[{task.name}] {algo} eval: {mean_r:.2f} +/- {std_r:.2f}")

    task_fname = safe_name(task.name)
    model_path = os.path.join(out_dir, f"{task_fname}_{algo}.zip")
    model.save(model_path)

    vec_path = None
    if isinstance(venv, VecNormalize):
        vec_path = os.path.join(out_dir, f"{task_fname}_{algo}_vecnormalize.pkl")
        venv.save(vec_path)

    venv.close()
    return {"task": task.name, "algo": algo, "mean": mean_r, "std": std_r, "model_path": model_path, "vec_path": vec_path}




def task_halfcheetah_target_velocity_render(target_v: float, seed: int = 0,
                                            vel_scale: float = 1.0, ctrl_cost_weight: float = 0.1):
    env = gym.make("HalfCheetah-v4", render_mode="human")
    env = TargetVelocity(env, target_velocity=target_v, vel_scale=vel_scale, ctrl_cost_weight=ctrl_cost_weight)
    env.reset(seed=seed)
    return env


def make_render_vecenv(task_make_env, seed=0):
    def _init():
        env = task_make_env()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return DummyVecEnv([_init])



def test_teacher_render(model_path: str, vec_path: Optional[str] = None, task: float = 1.0):
    # build render env (DummyVecEnv -> VecNormalize)
    venv = make_render_vecenv(lambda: task_halfcheetah_target_velocity_render(task, seed=0), seed=0)
    venv = VecNormalize.load(vec_path, venv)
    venv.training = False
    venv.norm_reward = False

    # load model
    model = SAC.load(model_path, env=venv)

    # rollout
    obs = venv.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        if done[0]:
            obs = venv.reset()

    venv.close()


def eval_teacher(model_path: str, vec_path: str, make_env_fn, n_eval_episodes: int = 10, seed: int = 0):
    def _init():
        env = make_env_fn()
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(vec_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = SAC.load(model_path, env=venv)

    mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=n_eval_episodes, deterministic=True)
    venv.close()
    return mean_r, std_r



