import math
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    x1 = X1 / lengthscale
    x2 = X2 / lengthscale
    x1_sq = (x1 ** 2).sum(dim=1).unsqueeze(1)
    x2_sq = (x2 ** 2).sum(dim=1).unsqueeze(0)
    sqdist = x1_sq + x2_sq - 2.0 * (x1 @ x2.t())
    K = variance * torch.exp(-0.5 * sqdist)
    return K


class NN_AGP_UCB_Adaptive:
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
        T_spec: int = 5,
        eps_prune: float = 1e-2,
    ):
        self.device = device or torch.device("cpu")
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.m = int(m)
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.jitter = jitter
        self.mll_steps = int(mll_steps)
        self.T_spec = T_spec
        self.eps_prune = eps_prune
        self.step_count = 0
        self.lr = lr

        self.encoder = NeuralEmbedding(theta_dim, self.m, hidden_dim).to(self.device)
        self.B = nn.Parameter(torch.randn(self.m, self.m, device=self.device) * 0.1)

        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(init_lengthscale, device=self.device)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(init_var, device=self.device)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(init_noise, device=self.device)))

        self._build_optimizer(lr)

        self.thetas: List[torch.Tensor] = []
        self.Xs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []

        self.m_history: List[int] = []

    def _build_optimizer(self, lr: float):
        params = list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise]
        self.optimizer = optim.Adam(params, lr=lr)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    @property
    def sigma_eps(self):
        return torch.exp(self.log_noise)


    def _assemble(self):
        if len(self.Xs) == 0:
            return None

        thetas_stack = torch.stack(self.thetas, dim=0).to(self.device)
        X = torch.stack(self.Xs, dim=0).to(self.device)
        y = torch.stack(self.ys, dim=0).view(-1,1).to(self.device)

        G = self.encoder(thetas_stack)

        Sigma_p = self.B @ self.B.t()
        k_x = rbf_kernel_matrix(X, X, self.lengthscale, self.variance)
        GSGT = (G @ Sigma_p) @ G.t()
        K_tilde = k_x * GSGT

        return {"G": G, "X": X, "y": y, "Sigma_p": Sigma_p, "K_tilde": K_tilde}


    def _neg_log_marginal_likelihood(self):
        mats = self._assemble()
        if mats is None:
            return None

        n = mats["K_tilde"].shape[0]
        K = mats["K_tilde"] + (self.sigma_eps ** 2 + self.jitter) * torch.eye(n, device=self.device)
        y = mats["y"]

        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K = K + 1e-6 * torch.eye(n, device=self.device)
            L = torch.linalg.cholesky(K)

        v = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.t(), v)

        logdet = 2.0 * torch.sum(torch.log(torch.diag(L) + 1e-12))
        mll = 0.5 * (y.t() @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)
        return mll.squeeze()


    def _fit_mll(self, steps: int = 10):
        if len(self.Xs) == 0:
            return

        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = self._neg_log_marginal_likelihood()
            if loss is None:
                break
            if not loss.requires_grad:
                print("DEBUG: loss.requires_grad=False")
                print("LOSS HAS NO GRAD!!!!!")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + [self.B, self.log_lengthscale, self.log_variance, self.log_noise],
                5.0
            )
            self.optimizer.step()


    def posterior(self, x: np.ndarray, theta: np.ndarray):
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
        g_t = self.encoder(theta_t).view(self.m,1)
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
            K = K_tilde + (self.sigma_eps ** 2 + self.jitter) * torch.eye(n, device=self.device)
            L = torch.linalg.cholesky(K)
            v = torch.linalg.solve(L, y)
            alpha = torch.linalg.solve(L.t(), v)

            k_x_vec = rbf_kernel_matrix(x_t, X, self.lengthscale, self.variance).view(-1)
            GS = G @ Sigma_p
            gts = (GS @ g_t).view(-1)
            k_tilde = k_x_vec * gts
            mu = (k_tilde.view(1,-1) @ alpha).squeeze()

            rhs = k_tilde.view(-1,1)
            v2 = torch.linalg.solve(L, rhs)
            prior_var = (g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x
            sigma2 = prior_var - (v2.t() @ v2).squeeze()
            sigma2 = torch.clamp(sigma2, min=1e-12)
            return float(mu.cpu().item()), float(sigma2.cpu().item())


    def select_action(self, X_candidates: np.ndarray, theta: np.ndarray):
        mus, sig2s = [], []
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
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).view(-1)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).view(-1)
        y_t = torch.tensor([y], dtype=torch.float32, device=self.device).view(-1)
        self.thetas.append(theta_t)
        self.Xs.append(x_t)
        self.ys.append(y_t)

        self._fit_mll(steps=10)
        self.step_count += 1

        self.m_history.append(self.m)

        if (self.step_count % self.T_spec != 0) or (len(self.thetas) < self.m):
            return

        G_stack = torch.stack(self.thetas, dim=0).to(self.device)
        with torch.no_grad():
            G_emb = self.encoder(G_stack)  # n x m_old
            U, S, Vh = torch.linalg.svd(G_emb, full_matrices=False)
            if S.numel() == 0:
                return
            m_star = int(torch.sum(S > self.eps_prune * S.max()).item())
            m_star = max(1, min(m_star, G_emb.shape[1]))
            if m_star >= G_emb.shape[1]:
                return

            V_top_raw = Vh[:m_star, :].to(self.device)

            B_old = self.B.data
            m_old = B_old.shape[0]
            if m_old != G_emb.shape[1]:
                B_tmp = torch.zeros((G_emb.shape[1], G_emb.shape[1]), device=self.device, dtype=B_old.dtype)
                min_dim = min(B_old.shape[0], B_tmp.shape[0])
                B_tmp[:min_dim, :min_dim] = B_old[:min_dim, :min_dim]
                B_old = B_tmp
                m_old = B_old.shape[0]

        B_new = V_top_raw @ B_old @ V_top_raw.t()
        self.B = nn.Parameter(B_new.to(self.device))

        new_encoder = NeuralEmbedding(self.theta_dim, m_star, self.hidden_dim).to(self.device)
        try:
            old_lin0 = self.encoder.net[0]
            new_lin0 = new_encoder.net[0]
            if old_lin0.weight.data.shape == new_lin0.weight.data.shape:
                new_lin0.weight.data[:] = old_lin0.weight.data.clone()
                new_lin0.bias.data[:] = old_lin0.bias.data.clone()
        except Exception:
            pass

        self.encoder = new_encoder
        self.m = m_star

        self._build_optimizer(self.lr)

        self._fit_mll(steps=10)


    def prior_variance(self, x: np.ndarray, theta: np.ndarray) -> float:
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
            g_t = self.encoder(theta_t).view(self.m,1)
            Sigma_p = self.B @ self.B.t()
            k_x_x = self.variance
            return float(((g_t.t() @ Sigma_p @ g_t).squeeze() * k_x_x).cpu().item())












