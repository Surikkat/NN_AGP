import math
from functools import lru_cache
from typing import List, Optional
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import torch
import torch.nn as nn
import torch.optim as optim


@lru_cache(maxsize=128)
def _mp_median(gamma: float) -> float:
    """Median of the nonzero unit-scale Marchenko--Pastur spectrum."""
    aspect = min(gamma, 1.0 / gamma)
    if aspect < 1e-12:
        return max(gamma, 1.0)
    lo, hi = (1 - math.sqrt(aspect)) ** 2, (1 + math.sqrt(aspect)) ** 2
    phi = (np.arange(100000) + 0.5) * (math.pi / 2) / 100000
    values = lo + (hi - lo) * np.sin(phi) ** 2
    cdf = np.cumsum((np.sin(phi) * np.cos(phi)) ** 2 / values)
    return max(gamma, 1.0) * float(np.interp(0.5, cdf / cdf[-1], values))


class NeuralEmbedding(nn.Module):
    def __init__(self, theta_dim: int, m: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


def rbf_kernel_matrix(X1: torch.Tensor, X2: torch.Tensor, lengthscale: torch.Tensor, variance: torch.Tensor):
    distances = torch.cdist(X1, X2).square()
    return variance * torch.exp(-distances / (2 * lengthscale ** 2))


class NN_AGP_UCB_Adaptive:
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        m: int = 100,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
        init_lengthscale: float = 1.0,
        init_var: float = 1.0,
        init_noise: float = 0.1,
        beta: Optional[float] = None,
        lr: float = 1e-3,
        mll_steps: int = 10,
        jitter: float = 1e-6,
        T_spec: int = 10,
        eps_prune: float = 1e-2,
        prune_rule: str = "relative",
        burn_in: int = 100,
        seed: Optional[int] = None,
    ):
        if prune_rule not in ("relative", "bbp", "none"):
            raise ValueError("prune_rule must be 'relative', 'bbp' or 'none'")
        if T_spec < 1 or burn_in < 0 or not 0 < eps_prune < 1:
            raise ValueError("Require T_spec >= 1, burn_in >= 0 and 0 < eps_prune < 1")
        self.device = device or torch.device("cpu")
        self.seed = torch.initial_seed() if seed is None else seed
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng([self.seed, 17])
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.m = int(m)
        self.m_max = self.m
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.jitter = jitter
        self.mll_steps = int(mll_steps)
        self.T_spec = T_spec
        self.eps_prune = eps_prune
        self.prune_rule = prune_rule
        self.burn_in = burn_in
        self.step_count = 0
        self.lr = lr

        self.encoder = NeuralEmbedding(theta_dim, self.m, hidden_dim).double().to(self.device)
        self.B = nn.Parameter((0.1 * torch.randn(self.m, self.m, dtype=torch.float64)).to(self.device))
        self.projection = torch.eye(self.m, dtype=self.B.dtype, device=self.device)
        self.epsilon = None
        self.spectral_history = []

        self.hyp = nn.Parameter(torch.tensor([math.log(init_lengthscale), math.log(init_var),
            math.log(init_noise ** 2)], dtype=torch.float64, device=self.device))

        self._build_optimizer(lr)

        self.thetas: List[torch.Tensor] = []
        self.Xs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []
        self.D2 = torch.zeros((0, 0), dtype=torch.float64, device=self.device)

        self.m_history: List[int] = []

    def _build_optimizer(self, lr: float):
        params = list(self.encoder.parameters()) + [self.B, self.hyp]
        self.optimizer = optim.Adam(params, lr=lr)

    @property
    def log_lengthscale(self):
        return self.hyp[0]

    @property
    def log_variance(self):
        return self.hyp[1]

    @property
    def log_noise(self):
        return self.hyp[2] / 2

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    @property
    def sigma_eps(self):
        return torch.exp(self.log_noise)


    def _assemble(self, compressed: bool = False, indices=None):
        if len(self.Xs) == 0:
            return None

        thetas_stack = torch.stack(self.thetas, dim=0).to(self.device)
        X = torch.stack(self.Xs, dim=0).to(self.device)
        y = torch.stack(self.ys, dim=0).view(-1).to(self.device)
        distances = self.D2
        if indices is not None:
            thetas_stack, X, y = thetas_stack[indices], X[indices], y[indices]
            distances = distances[indices][:, indices]

        G = self.encoder(thetas_stack)

        H = G @ self.B
        if compressed:
            H = H @ self.projection
        k_x = self.variance * torch.exp(-distances / (2 * self.lengthscale ** 2))
        K_tilde = k_x * (H @ H.t())

        return {"H": H, "X": X, "y": y, "K_tilde": K_tilde}


    def _neg_log_marginal_likelihood(self, indices=None):
        mats = self._assemble(indices=indices)
        if mats is None:
            return None

        n = mats["K_tilde"].shape[0]
        K = mats["K_tilde"] + (self.hyp[2].exp() + self.jitter) * torch.eye(n, dtype=self.B.dtype, device=self.device)
        y = mats["y"]
        L = torch.linalg.cholesky(K)
        return 0.5 * y @ torch.cholesky_solve(y[:, None], L)[:, 0] + L.diagonal().log().sum()


    def _fit_mll(self, steps: int = 10):
        if len(self.Xs) == 0:
            return
        n = len(self.Xs)
        indices = self.rng.choice(n, 256, replace=False) if n > 256 else np.arange(n)
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = self._neg_log_marginal_likelihood(indices)
            if loss is None:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 5.0)
            self.optimizer.step()
            with torch.no_grad():
                self.hyp[0].clamp_(math.log(0.05), math.log(20.0))
                self.hyp[1].clamp_(math.log(1e-3), math.log(100.0))
                self.hyp[2].clamp_(math.log(1e-6), math.log(100.0))


    def _posterior(self, X: np.ndarray, theta: np.ndarray):
        with torch.no_grad():
            Xq = torch.as_tensor(X, dtype=self.B.dtype, device=self.device)
            theta_t = torch.as_tensor(theta, dtype=self.B.dtype, device=self.device)[None]
            q = (self.encoder(theta_t) @ self.B @ self.projection).cpu().numpy()[0]
            ell, scale, noise = self.hyp.exp().cpu().numpy()
            prior = float(scale * (q @ q))
            if not self.Xs:
                return torch.zeros(len(X), dtype=self.B.dtype), torch.full((len(X),), prior, dtype=self.B.dtype)
            contexts, history = torch.stack(self.thetas), torch.stack(self.Xs)
            h = (self.encoder(contexts) @ self.B @ self.projection).cpu().numpy()
            y = torch.stack(self.ys).reshape(-1).cpu().numpy()
            K = scale * torch.exp(-self.D2 / (2 * ell ** 2)).cpu().numpy() * (h @ h.T)
            distances = torch.cdist(Xq, history).square()
            cross = scale * torch.exp(-distances / (2 * ell ** 2)).cpu().numpy() * (h @ q)[None]
            L = cholesky(K + (noise + self.jitter) * np.eye(len(K)), lower=True, check_finite=False)
            alpha = cho_solve((L, True), y, check_finite=False)
            forward = solve_triangular(L, cross.T, lower=True, check_finite=False)
            mean = cross @ alpha
            var = np.maximum(prior - (forward ** 2).sum(0), 1e-12)
            return torch.as_tensor(mean, dtype=self.B.dtype), torch.as_tensor(var, dtype=self.B.dtype)

    def posterior(self, x: np.ndarray, theta: np.ndarray):
        mean, var = self._posterior(np.asarray(x)[None], theta)
        return float(mean[0]), float(var[0])

    def select_action(self, X_candidates: np.ndarray, theta: np.ndarray):
        mean, var = self._posterior(X_candidates, theta)
        beta = self.beta
        if beta is None:
            beta = 2 * math.log(len(X_candidates) * (self.step_count + 1) ** 2 * math.pi ** 2 / (2 * 0.1))
        idx = int(torch.argmax(mean + math.sqrt(beta) * var.sqrt()))
        return X_candidates[idx], float(mean[idx]), float(var[idx])


    def update(self, theta: np.ndarray, x: np.ndarray, y: float,
               calibration_contexts: Optional[np.ndarray] = None):
        theta_t = torch.tensor(theta, dtype=self.B.dtype, device=self.device).view(-1)
        x_t = torch.tensor(x, dtype=self.B.dtype, device=self.device).view(-1)
        y_t = torch.tensor([y], dtype=self.B.dtype, device=self.device).view(-1)
        old_x = torch.stack(self.Xs) if self.Xs else x_t.new_empty((0, self.x_dim))
        distances = torch.cdist(old_x, x_t[None]).square()
        self.D2 = torch.cat([torch.cat([self.D2, distances], 1),
            torch.cat([distances.T, x_t.new_zeros((1, 1))], 1)], 0)
        self.thetas.append(theta_t)
        self.Xs.append(x_t)
        self.ys.append(y_t)

        self._fit_mll(steps=self.mll_steps)
        self.step_count += 1

        self.m_history.append(self.m)

        if (self.prune_rule == "none" or self.step_count < self.burn_in
                or self.step_count % self.T_spec != 0):
            return
        self._update_projection(calibration_contexts)

    def _update_projection(self, calibration_contexts=None):
        """Select GP directions from the current joint covariance."""
        contexts = (torch.stack(self.thetas) if calibration_contexts is None else
                    torch.as_tensor(calibration_contexts, dtype=self.B.dtype, device=self.device))
        if contexts.ndim != 2 or contexts.shape[0] == 0 or contexts.shape[1] != self.theta_dim:
            raise ValueError("Calibration contexts must have shape (n, theta_dim), n > 0")
        with torch.no_grad():
            H = (self.encoder(contexts) @ self.B).cpu().numpy()
            n = len(H)
            background = None
            if self.prune_rule == "relative":
                _, singular, vectors = np.linalg.svd(H, full_matrices=False)
                values = singular * singular / n
                cutoff = self.eps_prune * values[0]
                self.m = max(1, int(np.sum(values > cutoff)))
                projection = vectors[:self.m].T
                self.epsilon = self.eps_prune
                d = self.m_max
            else:
                B = self.B.cpu().numpy()
                last = self.encoder.net[-1]
                W, bias = last.weight.cpu().numpy(), last.bias.cpu().numpy()
                support = np.column_stack((B.T @ W, B.T @ bias))
                Q, s, _ = np.linalg.svd(support, full_matrices=False)
                Q = Q[:, s > np.finfo(float).eps * max(support.shape) * s[0]]
                h = H @ Q
                d = h.shape[1]
                values, vectors = np.linalg.eigh(h.T @ h / n)
                values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
                cutoff, proposed = None, 0
                if d:
                    gamma = d / n
                    background = float(np.median(values[:min(d, n)])) / _mp_median(gamma)
                    cutoff = background * (1 + np.sqrt(gamma)) ** 2
                    if np.isfinite(background) and background > np.finfo(float).eps * values[0]:
                        proposed = int(np.sum(values > cutoff))
                self.m = proposed or self.m_max
                projection = Q @ vectors[:, :proposed] if proposed else np.eye(self.m_max)
                self.epsilon = float(cutoff / values[0]) if d and values[0] > 0 else None
            self.projection = torch.as_tensor(projection.copy(), dtype=self.B.dtype, device=self.device)
            self.spectral_history.append(dict(t=self.step_count, n=n, dimension=d,
                rank=self.m, cutoff=cutoff, epsilon=self.epsilon, background=background))


    def prior_variance(self, x: np.ndarray, theta: np.ndarray) -> float:
        with torch.no_grad():
            theta_t = torch.as_tensor(theta, dtype=self.B.dtype, device=self.device).unsqueeze(0)
            h_t = self.encoder(theta_t) @ self.B @ self.projection
            return float((h_t.square().sum() * self.variance).cpu().item())
