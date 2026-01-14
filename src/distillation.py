from typing import Callable, Dict, Any, Tuple, List
import os
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from src.teacher import *
from src.memory import load_sac_teacher, collect_memory_from_sac_teacher, save_memory_npz

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GaussianStudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_bounds

        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)

    def forward(self, obs, return_features=False):
        z = self.backbone(obs)
        mu = self.mu_head(z)
        log_std = torch.clamp(self.log_std_head(z), self.log_std_min, self.log_std_max)
        if return_features:
            return mu, log_std, z
        return mu, log_std

class DistillMemoryDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)

        self.obs = d["obs"].astype(np.float32)
        self.mu_t = d["mu"].astype(np.float32)
        self.log_std_t = d["log_std"].astype(np.float32)

        self.action_t = (
            d["action"].astype(np.float32)
            if "action" in d.files else None
        )

        self.task = d["task"].item() if "task" in d.files else None
        self.task_id = int(d["task_id"]) if "task_id" in d.files else None

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx])
        mu_t = torch.from_numpy(self.mu_t[idx])
        log_std_t = torch.from_numpy(self.log_std_t[idx])

        if self.action_t is None:
            return obs, mu_t, log_std_t

        action_t = torch.from_numpy(self.action_t[idx])
        return obs, mu_t, log_std_t, action_t


class ReplayPool:
    """
    Keeps a bounded subset of samples from previous tasks in RAM for fast replay.
    """
    def __init__(self, max_per_task: int = 50_000):
        self.max_per_task = int(max_per_task)
        self.obs = []
        self.mu = []
        self.log_std = []
        self.action = []
        self.has_action = None

    def __len__(self):
        return int(sum(x.shape[0] for x in self.obs))

    def add_npz(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)

        obs = d["obs"].astype(np.float32)
        mu  = d["mu"].astype(np.float32)
        ls  = d["log_std"].astype(np.float32)

        has_action = ("action" in d.files)
        if self.has_action is None:
            self.has_action = has_action
        elif self.has_action != has_action:
            raise ValueError("ReplayPool: mixed presence of 'action' across npz files.")

        if obs.shape[0] > self.max_per_task:
            idx = np.random.choice(obs.shape[0], self.max_per_task, replace=False)
            obs = obs[idx]
            mu  = mu[idx]
            ls  = ls[idx]
            if has_action:
                act = d["action"].astype(np.float32)[idx]
        else:
            if has_action:
                act = d["action"].astype(np.float32)

        self.obs.append(torch.from_numpy(obs))       
        self.mu.append(torch.from_numpy(mu))
        self.log_std.append(torch.from_numpy(ls))
        if has_action:
            self.action.append(torch.from_numpy(act))

    def sample(self, batch_size: int):
        if len(self.obs) == 0:
            return None

        k = np.random.randint(0, len(self.obs))
        N = self.obs[k].shape[0]
        idx = torch.randint(0, N, (batch_size,))

        obs = self.obs[k][idx]
        mu  = self.mu[k][idx]
        ls  = self.log_std[k][idx]

        if self.has_action:
            act = self.action[k][idx]
            return obs, mu, ls, act
        return obs, mu, ls


# D1
def diag_gaussian_kl(mu_t, log_std_t, mu_s, log_std_s):
    std_t = torch.exp(log_std_t)
    std_s = torch.exp(log_std_s)

    var_t = std_t ** 2
    var_s = std_s ** 2

    kl = (log_std_s - log_std_t) + (var_t + (mu_t - mu_s) ** 2) / (2.0 * var_s) - 0.5
    return kl.sum(dim=-1).mean()  

# D2
def action_mse(mu_s, action_t, action_space=None):
    action = squash(mu_s, action_space)  
    return F.mse_loss(action, action_t)

# D3
def certainty_weights(log_std_t, eps=1e-6):
    # weight per sample (B,)
    std_t = torch.exp(log_std_t)             
    w = 1.0 / (eps + std_t.mean(dim=-1))      
    w = w / (w.mean() + 1e-8)
    return w

def weighted_diag_gaussian_kl(mu_t, log_std_t, mu_s, log_std_s):
    std_t = torch.exp(log_std_t)
    std_s = torch.exp(log_std_s)
    var_t = std_t ** 2
    var_s = std_s ** 2

    kl_per_dim = (log_std_s - log_std_t) + (var_t + (mu_t - mu_s) ** 2) / (2.0 * var_s) - 0.5
    kl_per_sample = kl_per_dim.sum(dim=-1)  

    w = certainty_weights(log_std_t)         
    return (w * kl_per_sample).mean()


@torch.no_grad()
def sac_teacher_latent(model, obs_batch_np):
    """
    Returns teacher actor latent features for given (normalized) obs batch.
    obs_batch_np: (B, obs_dim) or (1, obs_dim)
    """
    actor = model.policy.actor
    obs = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=model.device)

    if hasattr(actor, "features_extractor") and hasattr(actor, "latent_pi"):
        feat = actor.features_extractor(obs)
        lat = actor.latent_pi(feat)
        return lat

    if hasattr(actor, "extract_features") and hasattr(actor, "latent_pi"):
        feat = actor.extract_features(obs)
        lat = actor.latent_pi(feat)
        return lat

    raise AttributeError("Could not locate actor latent pathway. Inspect model.policy.actor to adapt extractor.")

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)


def latent_cosine_loss(z_t, z_s_proj):
    # (B, D): 1 - cosine similarity
    return 1.0 - F.cosine_similarity(z_s_proj, z_t, dim=-1).mean()


def snapshot_params(model: torch.nn.Module):
    """Detached copy of all trainable parameters (for anchoring)."""
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]

def anchor_loss(model: torch.nn.Module, anchor_params, coeff: float):
    """L2 penalty to keep parameters close to anchor snapshot."""
    if coeff <= 0 or anchor_params is None:
        return 0.0

    loss = 0.0
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            loss = loss + torch.sum((p - anchor_params[i]) ** 2)
            i += 1
    return coeff * loss


def squash(mu, action_space):
    a = torch.tanh(mu)
    low = torch.as_tensor(action_space.low, device=mu.device)
    high = torch.as_tensor(action_space.high, device=mu.device)
    return (low + (a + 1) * 0.5 * (high - low))

def sample_action_from_student(
    student,
    obs: torch.Tensor,           # shape: (B, obs_dim)
    low,
    high,                # gym.spaces.Box
    deterministic: bool = False,
):
    """
    Returns:
      action: (B, act_dim) in env action space
      pre_tanh_action: (B, act_dim) before tanh
    """
    mu, log_std = student(obs)
    std = torch.exp(log_std)

    if deterministic:
        pre_tanh = mu
    else:
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps   # reparameterization

    # squash
    tanh_action = torch.tanh(pre_tanh)

    # scale to env bounds
    action = low + 0.5 * (tanh_action + 1.0) * (high - low)

    return action, pre_tanh

def train_distill_step_with_replay(
    student,
    method: str,
    current_npz: str,
    replay_pool: ReplayPool = None,   
    replay_ratio: float = 0.10,
    teacher_sac_model=None,
    projector=None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    lambda_feat: float = 0.05,
    anchor_coeff: float = 1e-6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    method = method.upper()
    replay_pool = replay_pool  

    # current data loader
    ds_cur = DistillMemoryDataset(current_npz)
    dl_cur = DataLoader(ds_cur, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # batch split
    has_replay = (replay_pool is not None and len(replay_pool) > 0 and replay_ratio > 0)
    bs_rep = int(round(batch_size * replay_ratio)) if has_replay else 0
    bs_rep = max(0, min(batch_size - 1, bs_rep)) if batch_size > 1 else 0
    bs_cur = batch_size - bs_rep

    student = student.to(device)
    anchor_params = snapshot_params(student) if anchor_coeff > 0 else None

    # cache action scaling tensors once (speeds up D2)
    if teacher_sac_model is not None:
        act_space = teacher_sac_model.get_env().action_space
        low_t  = torch.as_tensor(act_space.low,  device=device, dtype=torch.float32)
        high_t = torch.as_tensor(act_space.high, device=device, dtype=torch.float32)
    else:
        low_t = high_t = None

    def squash_cached(mu):
        a = torch.tanh(mu)
        return (low_t + (a + 1.0) * 0.5 * (high_t - low_t))

    # projector setup
    if method == "D4_KL_LATENT":
        if teacher_sac_model is None:
            raise ValueError("D4_KL_LATENT requires teacher_sac_model.")
        if projector is None:
            sample_obs = ds_cur.obs[:8].astype(np.float32)
            with torch.no_grad():
                obs_t = torch.as_tensor(sample_obs, dtype=torch.float32, device=device)
                _, _, z_s = student(obs_t, return_features=True)
                student_lat_dim = int(z_s.shape[-1])
                obs_teacher = sample_obs[:, :obs_t.shape[-1]-3]
                z_t = sac_teacher_latent(teacher_sac_model, obs_teacher).detach().cpu()
                teacher_lat_dim = int(z_t.shape[-1])

            projector = Projector(student_lat_dim, teacher_lat_dim).to(device)
        else:
            projector = projector.to(device)

        opt = torch.optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=lr)
    else:
        opt = torch.optim.Adam(student.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        losses = []

        for batch in dl_cur:
            # shrink current batch
            if bs_rep > 0:
                batch = tuple(x[:bs_cur] for x in batch)

            if len(batch) == 3:
                obs_c, mu_t_c, log_std_t_c = batch
                action_t_c = None
            else:
                obs_c, mu_t_c, log_std_t_c, action_t_c = batch

            # sample replay
            if bs_rep > 0:
                rep = replay_pool.sample(bs_rep)
                rep = tuple(x.to(device) for x in rep)
            else:
                rep = None

            # move current to device
            obs_c = obs_c.to(device); mu_t_c = mu_t_c.to(device); log_std_t_c = log_std_t_c.to(device)
            if action_t_c is not None:
                action_t_c = action_t_c.to(device)

            # merge
            if rep is not None:
                if len(rep) == 3:
                    obs_r, mu_t_r, log_std_t_r = rep
                    action_t_r = None
                else:
                    obs_r, mu_t_r, log_std_t_r, action_t_r = rep

                obs = torch.cat([obs_c, obs_r], dim=0)
                mu_t = torch.cat([mu_t_c, mu_t_r], dim=0)
                log_std_t = torch.cat([log_std_t_c, log_std_t_r], dim=0)
                if action_t_c is not None:
                    action_t = torch.cat([action_t_c, action_t_r], dim=0)
                else:
                    action_t = None
            else:
                obs, mu_t, log_std_t, action_t = obs_c, mu_t_c, log_std_t_c, action_t_c

            opt.zero_grad()

            
            if method == "D4_KL_LATENT":
                mu_s, log_std_s, z_s = student(obs, return_features=True)
            else:
                mu_s, log_std_s = student(obs)

    
            if method == "D1_KL":
                loss = diag_gaussian_kl(mu_t, log_std_t, mu_s, log_std_s)

            elif method == "D2_MSE":
                if action_t is None:
                    raise ValueError("D2_MSE needs 'action' stored in npz.")
                if teacher_sac_model is None:
                    raise ValueError("D2_MSE needs teacher_sac_model (action scaling).")
                # Create the gaussian out of mu_s and squash it
                action, _ = sample_action_from_student(student, obs, low_t, high_t, deterministic=False)
                loss = F.mse_loss(action, action_t)

            elif method == "D3_WKL":
                loss = weighted_diag_gaussian_kl(mu_t, log_std_t, mu_s, log_std_s)

            elif method == "D4_KL_LATENT":
                loss_policy = diag_gaussian_kl(mu_t, log_std_t, mu_s, log_std_s)
                obs_teacher = obs.detach().cpu().numpy()[:, :obs.shape[-1]-3]
                z_t = sac_teacher_latent(teacher_sac_model, obs_teacher).to(device)
                z_s_proj = projector(z_s)
                loss_lat = latent_cosine_loss(z_t, z_s_proj)
                loss = loss_policy + lambda_feat * loss_lat

            else:
                raise ValueError("Unknown method.")

            if anchor_coeff > 0:
                loss = loss + anchor_loss(student, anchor_params, anchor_coeff)

            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        if ep == 1 or ep % 10 == 0:
            extra = f" (lambda_feat={lambda_feat})" if method == "D4_KL_LATENT" else ""
            print(f"Epoch {ep:02d} | {method} loss: {np.mean(losses):.4f} | replay={replay_ratio:.2f} | anchor={anchor_coeff}{extra}")

    return student, projector


def training_loop_with_replay(student, 
                              projector, 
                              method, 
                              TASK_SEQUENCE,
                              epochs,
                              max_replay_per_task: int = 60_000,
                              replay_ratio: float = 0.20,
                              anchor_coeff: float = 1e-4):
    replay_pool = ReplayPool(max_per_task=max_replay_per_task)

    for i, cfg in enumerate(TASK_SEQUENCE):
        print(f"\n==============================")
        print(f" Training task {i+1}/{len(TASK_SEQUENCE)}: {cfg['name']}")
        print(f"==============================")
        # Build Task object directly from env_fn
        task = Task(cfg["name"], cfg["env_fn"])

        # load teacher (same)
        if method == "D4_KL_LATENT" or method == "D2_MSE":
            teacher_model, _ = load_sac_teacher(
                task,
                cfg["model_path"],
                cfg["vec_path"],
                seed=0
            )

            student, projector = train_distill_step_with_replay(
                student=student,
                projector=projector,         
                method=method,
                current_npz=cfg["npz_path"],
                replay_pool=replay_pool,    
                replay_ratio=replay_ratio,          
                teacher_sac_model=teacher_model,
                epochs=epochs,
                lambda_feat=0.2,
                anchor_coeff=anchor_coeff,
            )
        else:
            student, projector = train_distill_step_with_replay(
                student=student,
                projector=projector,
                method=method,
                current_npz=cfg["npz_path"],
                replay_pool=replay_pool,  
                replay_ratio=replay_ratio,    
                epochs=epochs,
                anchor_coeff=anchor_coeff,
            )

        replay_pool.add_npz(cfg["npz_path"])

        venv_eval = build_vec_env(task, seed=0, normalize_obs=False)
        if cfg["vec_path"] is not None:
            venv_eval = VecNormalize.load(cfg["vec_path"], venv_eval)
            venv_eval.training = False
            venv_eval.norm_reward = False

        venv_eval.task_id = cfg["task_id"]
        venv_eval.n_tasks = cfg["n_tasks"]

        mean_ret, std_ret = eval_offline_student(student, venv_eval)
        print(f"--> Eval after task {i+1}: mean return = {mean_ret:.2f} +/- {std_ret:.2f}")
        venv_eval.close()
    
    return student, projector


def make_base_vec_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return DummyVecEnv([_init])

def load_eval_env_with_vecnorm(env_id: str, vec_path, informingvec_path: str, seed: int = 0,
                               task_id: int = 0, n_tasks: int = 1):
    """
    Loads eval env + VecNormalize stats and attaches task_id/n_tasks
    so eval_offline_student can auto-augment observations.
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([_init])

    if vec_path is not None:
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False

    venv.task_id = int(task_id)
    venv.n_tasks = int(n_tasks)

    return venv

@torch.no_grad()
def eval_offline_student(student, venv_eval, n_episodes: int = 10):
    """
    Evaluates student on venv_eval.
    Automatically appends task-ID one-hot to obs if student expects extra dims.

    Requires:
      venv_eval.task_id and venv_eval.n_tasks (set in load_eval_env_with_vecnorm)
    """
    device = next(student.parameters()).device
    student.eval()

    obs = venv_eval.reset()   
    if isinstance(obs, tuple):
        obs = obs[0]

    obs_dim_env = obs.shape[1]

    # infer student expected obs dim
    first_linear = None
    for m in student.modules():
        if isinstance(m, torch.nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("Could not find a Linear layer to infer student obs dim.")
    obs_dim_student = int(first_linear.in_features)

    extra = obs_dim_student - obs_dim_env
    if extra < 0:
        raise ValueError(f"Student expects obs_dim={obs_dim_student}, but env provides obs_dim={obs_dim_env}.")

    task_id = int(getattr(venv_eval, "task_id", 0))
    n_tasks = int(getattr(venv_eval, "n_tasks", extra if extra > 0 else 1))

    ep_returns = []
    cur_ret = 0.0

    while len(ep_returns) < n_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if extra > 0:
            tid = torch.zeros((obs_t.shape[0], extra), device=device, dtype=obs_t.dtype)
            if not (0 <= task_id < extra):
                raise ValueError(f"task_id={task_id} out of range for task one-hot length={extra}")
            tid[:, task_id] = 1.0
            obs_t = torch.cat([obs_t, tid], dim=1)

        mu_s, log_std_s = student(obs_t)
        action = squash(mu_s, venv_eval.action_space).cpu().numpy()

        obs, reward, done, info = venv_eval.step(action)

        cur_ret += float(reward[0])

        if bool(done[0]):
            ep_returns.append(cur_ret)
            cur_ret = 0.0
            obs = venv_eval.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    ep_returns = np.array(ep_returns, dtype=np.float32)
    return float(ep_returns.mean()), float(ep_returns.std())



