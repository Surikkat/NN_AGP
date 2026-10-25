# bandits/nn_agp.py
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DTYPE = torch.float64


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


def rbf_kernel_matrix(X1: torch.Tensor, X2: torch.Tensor,
                      lengthscale: torch.Tensor, variance: torch.Tensor):
    x1 = X1 / lengthscale
    x2 = X2 / lengthscale
    x1_sq = (x1 ** 2).sum(dim=1).unsqueeze(1)
    x2_sq = (x2 ** 2).sum(dim=1).unsqueeze(0)
    sqdist = x1_sq + x2_sq - 2.0 * (x1 @ x2.t())
    K = variance * torch.exp(-0.5 * sqdist)
    return K


class NN_AGP_UCB:
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        m: int = 5,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
        init_lengthscale: float = 1.0,
        init_var: float = 1.0,
        init_noise: float = 0.05,
        beta: float = 2.0,
        lr: float = 1e-3,
        mll_steps: int = 60,
        jitter: float = 1e-6,
        dtype: torch.dtype = DTYPE,
        batched: bool = False,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.batched = batched
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.m = int(m)
        self.beta = float(beta)
        self.jitter = float(jitter)
        self.mll_steps = int(mll_steps)

        self.encoder = NeuralEmbedding(theta_dim, self.m, hidden_dim).to(
            device=self.device, dtype=self.dtype
        )

        self.B = nn.Parameter(
            (torch.randn(self.m, self.m, device=self.device, dtype=self.dtype) * 0.1)
        )

        self.log_lengthscale = nn.Parameter(
            torch.log(torch.tensor(init_lengthscale, device=self.device, dtype=self.dtype))
        )
        self.log_variance = nn.Parameter(
            torch.log(torch.tensor(init_var, device=self.device, dtype=self.dtype))
        )
        self.log_noise = nn.Parameter(
            torch.log(torch.tensor(init_noise, device=self.device, dtype=self.dtype))
        )

        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + [self.B, self.log_lengthscale, self.log_variance, self.log_noise],
            lr=lr,
        )

        self.thetas: List[torch.Tensor] = []
        self.Xs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def sigma_eps(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    def _assemble(self):
        if len(self.Xs) == 0:
            return None

        thetas_stack = torch.stack(self.thetas, dim=0).to(self.device)
        X = torch.stack(self.Xs, dim=0).to(self.device)
        y = torch.stack(self.ys, dim=0).view(-1, 1).to(self.device)

        G = self.encoder(thetas_stack)

        Sigma_p = self.B @ self.B.t()
        k_x = rbf_kernel_matrix(X, X, self.lengthscale, self.variance)
        GS = G @ Sigma_p
        GSGT = GS @ G.t()
        K_tilde = k_x * GSGT

        return {
            "G": G,
            "X": X,
            "y": y,
            "Sigma_p": Sigma_p,
            "k_x": k_x,
            "GSGT": GSGT,
            "K_tilde": K_tilde,
        }

    def _neg_log_marginal_likelihood(self):
        mats = self._assemble()
        if mats is None:
            return None

        n = mats["K_tilde"].shape[0]
        K_base = mats["K_tilde"]
        y = mats["y"]

        jitter = float(self.jitter)
        success = False
        L = None

        attempts = 0
        while attempts < 6:
            try:
                K_try = K_base + (self.sigma_eps ** 2 + jitter) * torch.eye(
                    n, device=self.device, dtype=self.dtype
                )
                K_try = 0.5 * (K_try + K_try.t())
                L = torch.linalg.cholesky(K_try)
                success = True
                break
            except RuntimeError:
                jitter *= 10.0
                attempts += 1

        if not success:
            K_sym = 0.5 * (K_base + K_base.t())
            eigvals, eigvecs = torch.linalg.eigh(K_sym)
            min_eig = max(
                jitter,
                float(torch.max(-eigvals.min(), torch.tensor(1e-12, device=eigvals.device, dtype=self.dtype))),
            )
            min_eig = max(min_eig, float(eigvals.max()) * 1e-8)
            eigvals_clipped = torch.clamp(eigvals, min=min_eig)
            K = (eigvecs * eigvals_clipped.unsqueeze(0)) @ eigvecs.t()
            K = K + (self.sigma_eps ** 2 + jitter) * torch.eye(n, device=self.device, dtype=self.dtype)
            K = 0.5 * (K + K.t())
            L = torch.linalg.cholesky(K)
            print(f"[nn_agp] Warning: spectral stabilization in _nll (jitter~{jitter:.1e})")

        v = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.t(), v)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L) + 1e-12))
        mll = 0.5 * (y.t() @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)
        return mll.squeeze()

    def _fit_mll(self, steps: Optional[int] = None):
        if len(self.Xs) == 0:
            return
        steps = steps or self.mll_steps
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = self._neg_log_marginal_likelihood()
            if loss is None:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + [self.B, self.log_lengthscale, self.log_variance, self.log_noise],
                5.0,
            )
            self.optimizer.step()

    def posterior(self, x: np.ndarray, theta: np.ndarray):
        x_t = torch.tensor(x, dtype=self.dtype, device=self.device).unsqueeze(0)
        theta_t = torch.tensor(theta, dtype=self.dtype, device=self.device).unsqueeze(0)
        g_t = self.encoder(theta_t).view(self.m, 1)
        Sigma_p = self.B @ self.B.t()
        k_x_x = self.variance

        if len(self.Xs) == 0:
            prior_var = (g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x
            return 0.0, float(prior_var.cpu().item())

        with torch.no_grad():
            mats = self._assemble()
            G = mats["G"]
            X = mats["X"]
            y = mats["y"]
            K_tilde = mats["K_tilde"]
            n = X.shape[0]

            K_base = K_tilde
            jitter = float(self.jitter)
            success = False
            L = None
            attempts = 0
            while attempts < 6:
                try:
                    K = K_base + (self.sigma_eps ** 2 + jitter) * torch.eye(
                        n, device=self.device, dtype=self.dtype
                    )
                    K = 0.5 * (K + K.t())
                    L = torch.linalg.cholesky(K)
                    success = True
                    break
                except RuntimeError:
                    jitter *= 10.0
                    attempts += 1

            if not success:
                K_sym = 0.5 * (K_base + K_base.t())
                eigvals, eigvecs = torch.linalg.eigh(K_sym)
                min_eig = max(
                    jitter,
                    float(torch.max(-eigvals.min(), torch.tensor(1e-12, device=eigvals.device, dtype=self.dtype))),
                )
                min_eig = max(min_eig, float(eigvals.max()) * 1e-8)
                eigvals_clipped = torch.clamp(eigvals, min=min_eig)
                K = (eigvecs * eigvals_clipped.unsqueeze(0)) @ eigvecs.t()
                K = K + (self.sigma_eps ** 2 + jitter) * torch.eye(n, device=self.device, dtype=self.dtype)
                K = 0.5 * (K + K.t())
                L = torch.linalg.cholesky(K)
                print(f"[nn_agp] Warning: spectral stabilization in posterior (jitter~{jitter:.1e})")

            v = torch.linalg.solve(L, y)
            alpha = torch.linalg.solve(L.t(), v)

            k_x_vec = rbf_kernel_matrix(x_t, X, self.lengthscale, self.variance).view(-1)
            GS = G @ Sigma_p
            gts = (GS @ g_t).view(-1)
            k_tilde = k_x_vec * gts

            mu = (k_tilde.view(1, -1) @ alpha).squeeze()
            rhs = k_tilde.view(-1, 1)
            v2 = torch.linalg.solve(L, rhs)
            prior_var = (g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x
            sigma2 = prior_var - (v2.t() @ v2).squeeze()
            sigma2 = torch.clamp(sigma2, min=1e-12)
            return float(mu.cpu().item()), float(sigma2.cpu().item())

    def select_action(self, X_candidates: np.ndarray, theta: np.ndarray, batched: Optional[bool] = None):
        batched = self.batched if batched is None else batched
        if batched:
            with torch.no_grad():
                context = torch.tensor(theta, dtype=self.dtype or torch.float32, device=self.device)[None]
                g = self.encoder(context).reshape(self.m, 1)
                Sigma = self.B @ self.B.T
                prior = (g.T @ Sigma @ g).squeeze() * self.variance
                if not self.Xs:
                    means = np.zeros(len(X_candidates))
                    variances = np.full(len(X_candidates), float(prior))
                else:
                    mats = self._assemble()
                    K = mats["K_tilde"] + (self.sigma_eps ** 2 + self.jitter) * torch.eye(
                        len(self.Xs), dtype=self.dtype, device=self.device
                    )
                    K = 0.5 * (K + K.T)
                    try:
                        factor = torch.linalg.cholesky(K)
                    except RuntimeError:
                        return self.select_action(X_candidates, theta, batched=False)
                    alpha = torch.cholesky_solve(mats["y"], factor)
                    X = torch.as_tensor(X_candidates, dtype=self.dtype or torch.float32, device=self.device)
                    kernels = rbf_kernel_matrix(X, mats["X"], self.lengthscale, self.variance)
                    cross = kernels * (mats["G"] @ Sigma @ g).reshape(1, -1)
                    means = (cross @ alpha).reshape(-1).cpu().numpy()
                    forward = torch.linalg.solve_triangular(factor, cross.T, upper=False)
                    variances = torch.clamp(prior - forward.square().sum(0), min=1e-12).cpu().numpy()
                idx = int(np.argmax(means + math.sqrt(self.beta) * np.sqrt(variances)))
                return X_candidates[idx], float(means[idx]), float(variances[idx])

        mus = []
        sig2s = []
        for x in X_candidates:
            mu, s2 = self.posterior(x, theta)
            mus.append(mu)
            sig2s.append(s2)
        mus = np.array(mus)
        sig2s = np.array(sig2s)
        ucbs = mus + math.sqrt(self.beta) * np.sqrt(sig2s)
        idx = int(np.argmax(ucbs))
        return X_candidates[idx], mus[idx], sig2s[idx]

    def update(self, theta: np.ndarray, x: np.ndarray, y: float):
        theta_t = torch.tensor(theta, dtype=self.dtype, device=self.device)
        x_t = torch.tensor(x, dtype=self.dtype, device=self.device)
        y_t = torch.tensor([y], dtype=self.dtype, device=self.device)

        self.thetas.append(theta_t)
        self.Xs.append(x_t)
        self.ys.append(y_t)

        self._fit_mll()

    def prior_variance(self, x: np.ndarray, theta: np.ndarray) -> float:
        with torch.no_grad():
            theta_t = torch.tensor(theta, dtype=self.dtype, device=self.device).unsqueeze(0)
            g_t = self.encoder(theta_t).view(self.m, 1)
            Sigma_p = self.B @ self.B.t()
            k_x_x = self.variance
            return float(((g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x).cpu().item())








        