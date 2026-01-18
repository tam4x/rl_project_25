from typing import Callable, Dict, Any, Tuple, List
import os
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from src.teacher import *
from src.memory import load_sac_teacher, collect_memory_from_sac_teacher, save_memory_npz
from src.distillation import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def save_student(
    student,
    out_path: str,
    obs_dim: int,
    act_dim: int,
    method: str,
    projector=None,
    extra: dict | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "student_state_dict": student.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "method": method,
    }

    if projector is not None:
        payload["projector_state_dict"] = projector.state_dict()

    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, out_path)
    print(f"Saved student to {out_path}")

def load_student(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    student = GaussianStudentPolicy(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
    ).to(device)

    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()

    

    return student, ckpt

def evaluate_student_on_all_tasks(student, TASK_SEQUENCE, seed=0, n_episodes=10):
    results = []
    for i, cfg in enumerate(TASK_SEQUENCE):
        task = Task(cfg["name"], cfg["env_fn"])

        venv_eval = build_vec_env(task, seed=seed, normalize_obs=False)
        if cfg["vec_path"] is not None:
            venv_eval = VecNormalize.load(cfg["vec_path"], venv_eval)
            venv_eval.training = False
            venv_eval.norm_reward = False

        venv_eval.task_id = cfg["task_id"]
        venv_eval.n_tasks = cfg["n_tasks"]

        mean_ret, std_ret = eval_offline_student(student, venv_eval, n_episodes=n_episodes)
        venv_eval.close()

        results.append((cfg["name"], mean_ret, std_ret))
        print(f"[FINAL EVAL] {cfg['name']}: {mean_ret:.2f} +/- {std_ret:.2f}")

    return results