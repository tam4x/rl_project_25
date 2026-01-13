import os
import numpy as np
import gymnasium as gym

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

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
        env = Monitor(env)
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
    mu_t, log_std_t, _ = model.policy.actor.get_action_dist_params(obs_t)

    mu = mu_t.detach().cpu().numpy()
    log_std = log_std_t.detach().cpu().numpy()
    return mu, log_std


def augment_obs_with_task_id(
    obs_batch: np.ndarray,
    task_id: int,
    n_tasks: int,
    encoding: str = "onehot",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    obs_batch: (n_envs, obs_dim)
    Returns augmented obs: (n_envs, obs_dim + extra)
    """
    if encoding == "onehot":
        if not (0 <= task_id < n_tasks):
            raise ValueError(f"task_id={task_id} out of range for n_tasks={n_tasks}")
        tid = np.zeros((obs_batch.shape[0], n_tasks), dtype=dtype)
        tid[:, task_id] = 1.0
        return np.concatenate([obs_batch.astype(dtype, copy=False), tid], axis=1)

    if encoding == "scalar":
        tid = np.full((obs_batch.shape[0], 1), float(task_id), dtype=dtype)
        return np.concatenate([obs_batch.astype(dtype, copy=False), tid], axis=1)

    raise ValueError(f"Unknown encoding='{encoding}'. Use 'onehot' or 'scalar'.")


def collect_memory_from_sac_teacher(
    model: SAC,
    venv,
    task_name: str,
    task_id: int,                 
    n_tasks: int,                
    task_id_encoding: str = "onehot",  
    n_steps: int = 100_000,
    deterministic_action: bool = True,
    store_actions: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Collects memory dataset:
      - obs_aug (obs + task-id encoding)  <-- for student
      - mu, log_std (teacher dist params)
      - (optional) action

    IMPORTANT: teacher inference uses the ORIGINAL obs (not augmented).
    NOTE: obs from VecNormalize-wrapped venv are already normalized.
    """
    venv.seed(seed)
    obs = venv.reset()

    obs_list = []
    mu_list = []
    logstd_list = []
    act_list = []

    for _ in range(n_steps):
        # Teacher uses original obs
        mu, log_std = sac_policy_params(model, obs)
        action, _ = model.predict(obs, deterministic=deterministic_action)

        # Student memory stores augmented obs
        obs_aug = augment_obs_with_task_id(
            obs_batch=obs,
            task_id=task_id,
            n_tasks=n_tasks,
            encoding=task_id_encoding,
        )

        obs_list.append(obs_aug.copy())
        mu_list.append(mu.copy())
        logstd_list.append(log_std.copy())
        if store_actions:
            act_list.append(action.copy())

        obs, reward, done, info = venv.step(action)
        if bool(done[0]):
            obs = venv.reset()

    data = {
        "task": task_name,
        "task_id": int(task_id),                 
        "n_tasks": int(n_tasks),                 
        "task_id_encoding": task_id_encoding,    
        "obs": np.concatenate(obs_list, axis=0),        # (n_steps, obs_dim + extra)
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
        task_id=data["task_id"],                       
        n_tasks=data["n_tasks"],                       
        task_id_encoding=data["task_id_encoding"],     
        obs=data["obs"],
        mu=data["mu"],
        log_std=data["log_std"],
        **({"action": data["action"]} if "action" in data else {}),
    )
    print("Saved memory:", out_path)
