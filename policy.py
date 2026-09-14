"""Behaviour-cloning policies, training loops, and evaluation functions."""

import numpy as np
import torch
import torch.nn as nn

from env2d import ACT_DIM, MAX_STEPS, OBS_DIM, Push2DVec, expert_action

DEVICE = "cpu"
# Small networks run faster and more predictably with one intra-op thread;
# independent seeds can then be parallelized without CPU oversubscription.
torch.set_num_threads(1)


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


class TreePolicy:
    """Adapter exposing a scikit-learn regressor as a policy."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        pred = self.model.predict(x.detach().cpu().numpy())
        return torch.tensor(pred, dtype=torch.float32)


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


def train_tree_bc(obs, act, seed=0, n_estimators=40, min_samples_leaf=2,
                  **_unused):
    """Train an extremely-randomized-tree regression policy."""
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=1.0,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(obs, act)
    return TreePolicy(model)


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
