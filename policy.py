"""Behaviour-cloning policy, training loop, and the two evaluations.

The model is deliberately small and identical across every run. The model is
not the independent variable here -- the data is.
"""

import numpy as np
import torch
import torch.nn as nn

from env2d import ACT_DIM, MAX_STEPS, OBS_DIM, Push2DVec, expert_action

DEVICE = "cpu"


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(obs, act, seed=0, epochs=30, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    policy = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(act, dtype=torch.float32)
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = ((policy(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy.eval()


@torch.no_grad()
def eval_open_loop(policy, obs, act):
    """MSE against expert actions on held-out CLEAN expert states.
    This is the cheap, offline metric -- no environment interaction."""
    pred = policy(torch.tensor(obs, dtype=torch.float32)).numpy()
    return float(((pred - act) ** 2).mean())


@torch.no_grad()
def eval_closed_loop(policy, n_episodes=200, seed=12345):
    """Actually run the policy. Fixed eval seed so every config faces an
    identical set of initial conditions."""
    env = Push2DVec(n_episodes, seed=seed)
    obs = env.reset()
    for _ in range(MAX_STEPS):
        act = policy(torch.tensor(obs, dtype=torch.float32)).numpy()
        obs, succ, done = env.step(np.clip(act, -1, 1))
        if done.all():
            break
    return float(succ.mean()), float(env.dist_to_goal().mean())


@torch.no_grad()
def eval_wrong_side_rate(policy, n_episodes=200, seed=12345):
    """Diagnostic: fraction of failed episodes where the policy pushed the
    block FARTHER from the goal than it started. This is the signature
    compounding-error failure -- approaching from the wrong side."""
    env = Push2DVec(n_episodes, seed=seed)
    obs = env.reset()
    start = env.dist_to_goal().copy()
    for _ in range(MAX_STEPS):
        act = policy(torch.tensor(obs, dtype=torch.float32)).numpy()
        obs, succ, done = env.step(np.clip(act, -1, 1))
        if done.all():
            break
    worse = (env.dist_to_goal() > start + 0.02) & ~succ
    return float(worse.mean())
