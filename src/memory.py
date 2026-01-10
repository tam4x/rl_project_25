import os
import numpy as np
import gymnasium as gym

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch


@dataclass(frozen=True)
class Task:
    name: str
    make_env: Callable[[], gym.Env]

def build_vec_env(task: Task, seed: int = 0, normalize_obs: bool = True):
    def _init():
        env = task.make_env()
        env = Monitor(env)  # logs episode returns/lengths
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([_init])

    if normalize_obs:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return venv

def load_sac_teacher(task: Task, model_path: str, vec_path: Optional[str], seed: int = 0):
    """
    Loads SAC model + VecNormalize stats (if provided) for correct obs normalization.
    Returns (model, venv) where venv is ready for inference/eval.
    """
    venv = build_vec_env(task, seed=seed, normalize_obs=False)

    if vec_path is not None:
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = SAC.load(model_path, env=venv)
    return model, venv


@torch.no_grad()
def sac_policy_params(model: SAC, obs_batch: np.ndarray):
    """
    obs_batch: (n_envs, obs_dim) in VecEnv format (here n_envs=1).
    Returns:
      mu:      (n_envs, act_dim)
      log_std: (n_envs, act_dim)
    """
    obs_t = torch.as_tensor(obs_batch).to(model.device)

    # SB3 SAC actor helper
    mu_t, log_std_t, _ = model.policy.actor.get_action_dist_params(obs_t)

    mu = mu_t.detach().cpu().numpy()
    log_std = log_std_t.detach().cpu().numpy()
    return mu, log_std

def collect_memory_from_sac_teacher(
    model: SAC,
    venv,
    task_name: str,
    n_steps: int = 100_000,
    deterministic_action: bool = True,
    store_actions: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Collects memory dataset: obs_norm, mu, log_std, (optional) action.
    NOTE: obs from VecNormalize-wrapped venv are already normalized.
    """
    venv.seed(seed)
    obs = venv.reset()

    obs_list = []
    mu_list = []
    logstd_list = []
    act_list = []

    for _ in range(n_steps):
        mu, log_std = sac_policy_params(model, obs)

        action, _ = model.predict(obs, deterministic=deterministic_action)

        obs_list.append(obs.copy())
        mu_list.append(mu.copy())
        logstd_list.append(log_std.copy())
        if store_actions:
            act_list.append(action.copy())

        obs, reward, done, info = venv.step(action)

        if bool(done[0]):
            obs = venv.reset()

    data = {
        "task": task_name,
        "obs": np.concatenate(obs_list, axis=0),        # (n_steps, obs_dim)
        "mu": np.concatenate(mu_list, axis=0),          # (n_steps, act_dim)
        "log_std": np.concatenate(logstd_list, axis=0), # (n_steps, act_dim)
    }
    if store_actions:
        data["action"] = np.concatenate(act_list, axis=0)

    return data

def save_memory_npz(data: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        task=data["task"],
        obs=data["obs"],
        mu=data["mu"],
        log_std=data["log_std"],
        **({"action": data["action"]} if "action" in data else {}),
    )
    print("Saved memory:", out_path)