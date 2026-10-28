import copy
import math
from functools import lru_cache
from typing import List, Optional
import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh, solve_triangular, svd
import torch
import torch.nn as nn
import torch.optim as optim


@lru_cache(maxsize=128)
def _mp_median(aspect: float) -> float:
    """Unit-scale Marchenko--Pastur median by midpoint quadrature."""
    if aspect < 1e-12:
        return 1.0
    lo, hi = (1 - math.sqrt(aspect)) ** 2, (1 + math.sqrt(aspect)) ** 2
    phi = (np.arange(100000) + 0.5) * (math.pi / 2) / 100000
    values = lo + (hi - lo) * np.sin(phi) ** 2
    cdf = np.cumsum((np.sin(phi) * np.cos(phi)) ** 2 / values)
    return float(np.interp(0.5, cdf / cdf[-1], values))


def _svd_errors(h, result):
    u, s, vt = result
    if not all(torch.isfinite(v).all() for v in result):
        return {"nonfinite_factors": True}
    tol = max(1e-10, 100 * torch.finfo(h.dtype).eps)
    scale = h.abs().max().clamp_min(torch.finfo(h.dtype).tiny)
    a, values = h / scale, s / scale
    energy = a.square().sum()
    denominator = energy.clamp_min(torch.finfo(h.dtype).tiny)
    checks = {
        "negative_singular_value": float(s.min()) < 0,
        "unordered_spectrum": bool((s[1:] > s[:-1]).any()),
        "left_orthogonality_error": float((u[:, :len(s)].T @ u[:, :len(s)] -
            torch.eye(len(s), dtype=h.dtype, device=h.device)).abs().max()),
        "right_orthogonality_error": float((vt @ vt.T -
            torch.eye(len(vt), dtype=h.dtype, device=h.device)).abs().max()),
        "relative_trace_error": float((values.square().sum() - energy).abs() / denominator),
        "relative_reconstruction_error": float(((u[:, :len(s)] * values) @ vt[:len(s)] - a).norm() /
                                               denominator.sqrt()),
    }
    return {key: value for key, value in checks.items() if value is True or
            (not isinstance(value, bool) and (not math.isfinite(value) or value > tol))}


def _checked_svd(h, full_matrices, event):
    if not torch.isfinite(h).all():
        raise RuntimeError("Nonfinite SVD input: numerical fallback is not permitted")
    result = torch.linalg.svd(h, full_matrices=full_matrices)
    bad = _svd_errors(h, result)
    if not bad:
        return result
    factors = svd(h.detach().cpu().numpy(),
                  full_matrices=bool(full_matrices and h.shape[0] < h.shape[1]),
                  lapack_driver="gesvd", check_finite=True)
    result = tuple(torch.from_numpy(v.copy()).to(dtype=h.dtype, device=h.device) for v in factors)
    invalid = _svd_errors(h, result)
    if invalid:
        raise RuntimeError(f"LAPACK gesvd returned an invalid decomposition: {invalid}")
    event.update(input_shape=list(h.shape), reason=list(bad),
                 original_errors={key: value if math.isfinite(value) else None for key, value in bad.items()},
                 backend="scipy.linalg.svd/lapack_gesvd")
    return result


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
        debug_spectral: bool = False,
    ):
        if prune_rule not in ("relative", "bbp", "none"):
            raise ValueError("prune_rule must be 'relative', 'bbp' or 'none'")
        if T_spec < 1 or burn_in < 0 or not 0 < eps_prune < 1:
            raise ValueError("Require T_spec >= 1, burn_in >= 0 and 0 < eps_prune < 1")
        self.device = device or torch.device("cpu")
        self.debug_spectral = debug_spectral
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
        self.w = torch.ones(self.m, dtype=self.B.dtype, device=self.device)
        self.epsilon = None
        self.spectral_history = []
        self.proposed_rank, self.no_peak = self.m, False
        self.method = {"none": "Full", "relative": "Relative cutoff 1e-2", "bbp": "BBP"}[prune_rule]
        self.calibration_mode, self.calibration = "auto", None
        self._spectral_backend = "support" if prune_rule == "bbp" else "numpy"
        self.decision_rank = None
        self.history_svd, self.corrected_svd = False, True
        self.fold_round, self._spectral_frozen = None, False
        self.transfer = None
        self.catalog = False
        self.catalog_diagnostics = {}
        self.numerical_fallbacks = []
        self.last_mean = self.last_var = None
        self.c = dict(M=self.m, hidden=[hidden_dim], noise_sd=init_noise, cov_lr=lr,
                      cov_steps=mll_steps, cov_mle_max=256, cov_start=burn_in,
                      T_spec=T_spec, eps=eps_prune, c=1.0, delta=0.1)
        self.cfg = dict(env=dict(d_theta=theta_dim, d_x=x_dim), agent=self.c)

        self.hyp = nn.Parameter(torch.tensor([math.log(init_lengthscale), math.log(init_var),
            math.log(init_noise ** 2)], dtype=torch.float64, device=self.device))

        self._build_optimizer(lr)

        self.thetas: List[torch.Tensor] = []
        self.Xs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []
        self.D2 = torch.zeros((0, 0), dtype=torch.float64, device=self.device)

        self.m_history: List[int] = []

    @classmethod
    def from_config(cls, cfg, seed, method, calibration="history", history_svd=False,
                    corrected_svd=True, fold_round=None, catalog=False):
        thresholds = {"Relative cutoff 1e-2": .01, "Relative cutoff 1e-4": .0001,
                      "Relative cutoff 1e-6": 1e-6, "Relative cutoff 1e-5": 1e-5}
        ranks = {"Fixed rank 25": 25, "Fixed rank 10": 10}
        if history_svd:
            ranks.update({"Fixed rank 3": 3, "Fixed rank 5": 5})
        if method not in ("Full", "BBP", *thresholds, *ranks):
            raise ValueError(f"Unknown adaptive method: {method}")
        if calibration not in ("history", "fresh"):
            raise ValueError("calibration must be 'history' or 'fresh'")
        if history_svd and (method == "BBP" or calibration != "history" or catalog):
            raise ValueError("history_svd requires history calibration and a Torch posterior; BBP/catalog are incompatible")
        e, c = cfg["env"], cfg["agent"]
        width = c["M"]
        decision_rank = c.get("decision_rank") if history_svd else None
        if decision_rank is None:
            decision_rank = ranks.get(method)
        if decision_rank is not None and (method == "Full" or type(decision_rank) is not int or
                                          not 1 <= decision_rank <= width):
            raise ValueError("decision_rank requires hard mode and an integer in [1, M]")
        start = c.get("cov_start", width if history_svd else 100)
        period = c["T_spec"]
        if fold_round is not None and (type(fold_round) is not int or fold_round < max(1, start) or
                                       fold_round % period or method != "BBP"):
            raise ValueError("fold_round must be a BBP spectral round at or after cov_start")
        epsilon = c.get("eps", .01) if history_svd and method == "Relative cutoff 1e-2" else thresholds.get(method, .01)
        obj = cls(e["d_theta"], e["d_x"], m=width, hidden_dim=c["hidden"][0],
                  init_noise=c.get("noise_sd", .1), lr=c.get("cov_lr", 1e-3),
                  mll_steps=c.get("cov_steps", 10), T_spec=period, eps_prune=epsilon,
                  prune_rule="none" if method == "Full" else "bbp" if method == "BBP" else "relative",
                  burn_in=start, seed=seed)
        obj.cfg, obj.c = cfg, c
        obj.method, obj.calibration_mode = method, calibration
        obj.history_svd, obj.corrected_svd = history_svd, corrected_svd
        obj.fold_round, obj.catalog = fold_round, catalog
        obj.decision_rank = decision_rank
        obj._spectral_backend = ("torch" if history_svd else "support" if
            method in ("BBP", "Relative cutoff 1e-2") or
            (method == "Relative cutoff 1e-4" and calibration == "history") else "numpy")
        obj.epsilon = .01 if method == "Relative cutoff 1e-2" and not history_svd else None
        return obj


    @property
    def th(self):
        return torch.stack(self.thetas) if self.thetas else self.B.new_empty((0, self.theta_dim))


    @property
    def x(self):
        return torch.stack(self.Xs) if self.Xs else self.B.new_empty((0, self.x_dim))


    @property
    def y(self):
        return torch.stack(self.ys).reshape(-1) if self.ys else self.B.new_empty(0)


    def features(self, th, compressed=False):
        h = self.encoder(th) @ self.B
        return (h @ self.projection) * self.w.sqrt() if compressed else h

    def kernel(self, h1, h2, distances):
        return self.hyp[1].exp() * torch.exp(-distances / (2 * self.hyp[0].exp() ** 2)) * (h1 @ h2.T)

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

        H = self.features(thetas_stack, compressed)
        K_tilde = self.kernel(H, H, distances)

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
        limit = self.c.get("cov_mle_max", 256)
        indices = self.rng.choice(n, limit, replace=False) if n > limit else np.arange(n)
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
        self.catalog_diagnostics = {}
        if self.history_svd:
            return self._torch_posterior(X, theta)
        if self.catalog and len(X) * self.m < len(self.ys):
            return self._catalog_posterior(X, theta)
        return self._dual_posterior(X, theta)

    def _dual_posterior(self, X, theta):
        with torch.no_grad():
            Xq = torch.as_tensor(X, dtype=self.B.dtype, device=self.device)
            theta_t = torch.as_tensor(theta, dtype=self.B.dtype, device=self.device)[None]
            h = self.features(self.th, True).cpu().numpy()
            q = self.features(theta_t, True).cpu().numpy()[0]
            ell, scale, noise = self.hyp.exp().cpu().numpy()
            prior = float(scale * (q @ q))
            if not self.Xs:
                return torch.zeros(len(X), dtype=self.B.dtype), torch.full((len(X),), prior, dtype=self.B.dtype)
            history, y = self.x, self.y.cpu().numpy()
            K = scale * torch.exp(-self.D2 / (2 * ell ** 2)).cpu().numpy() * (h @ h.T)
            distances = torch.cdist(Xq, history).square()
            cross = scale * torch.exp(-distances / (2 * ell ** 2)).cpu().numpy() * (h @ q)[None]
            L = cholesky(K + (noise + self.jitter) * np.eye(len(K)), lower=True, check_finite=False)
            alpha = cho_solve((L, True), y, check_finite=False)
            forward = solve_triangular(L, cross.T, lower=True, check_finite=False)
            mean = cross @ alpha
            var = np.maximum(prior - (forward ** 2).sum(0), 1e-12)
            return torch.as_tensor(mean, dtype=self.B.dtype), torch.as_tensor(var, dtype=self.B.dtype)

    def _torch_posterior(self, X, theta):
        with torch.no_grad():
            theta_t = torch.as_tensor(theta, dtype=self.B.dtype, device=self.device)[None]
            q = self.features(theta_t, True)
            prior = self.hyp[1].exp() * q.square().sum().expand(len(X))
            if not self.ys:
                return self.B.new_zeros(len(X)), prior
            h = self.features(self.th, True)
            distances = torch.cdist(torch.as_tensor(X, dtype=self.B.dtype, device=self.device), self.x).square()
            cross = self.kernel(q.expand(len(X), -1), h, distances)
            K = self.kernel(h, h, self.D2)
            L = torch.linalg.cholesky(K + (self.hyp[2].exp() + self.jitter) *
                                      torch.eye(len(K), dtype=K.dtype, device=K.device))
            forward = torch.linalg.solve_triangular(L, cross.T, upper=False)
            mean = cross @ torch.cholesky_solve(self.y[:, None], L)[:, 0]
            return mean, (prior - (forward ** 2).sum(0)).clamp_min(1e-12)

    def _catalog_posterior(self, X, theta):
        X = np.asarray(X)
        with torch.no_grad():
            H = self.features(self.th, True).cpu().numpy()
            q = self.features(torch.as_tensor(theta, dtype=self.B.dtype, device=self.device)[None], True).cpu().numpy()[0]
            ell, scale, noise = self.hyp.exp().cpu().numpy()
            y, observed = self.y.cpu().numpy(), self.x.cpu().numpy()
        K, m = len(X), H.shape[1]
        matches = np.all(observed[:, None, :] == X[None, :, :], axis=2)
        if not np.all(matches.sum(1) == 1):
            raise ValueError("History must use this fixed, unique catalog")
        ids = matches.argmax(1)
        distances = np.square(X[:, None, :] - X[None, :, :]).sum(2)
        Kx = scale * np.exp(-distances / (2 * ell ** 2))
        eigenvalues, vectors = eigh(Kx, check_finite=False)
        tolerance = 1e-12 * (1 + np.abs(Kx).max())
        if eigenvalues.min() < -tolerance:
            raise RuntimeError("Action kernel is not PSD within roundoff")
        R = vectors * np.sqrt(np.maximum(eigenvalues, 0))
        kernel_error = float(np.max(np.abs(R @ R.T - Kx)))
        if kernel_error > tolerance:
            raise RuntimeError("Action factorization changed the kernel")
        grams, response = np.zeros((K, m, m)), np.zeros((K, m))
        for j in range(K):
            group = H[ids == j]
            grams[j] = group.T @ group
            response[j] = group.T @ y[ids == j]
        eta = float(noise + self.jitter)
        products = (R[:, :, None] * R[:, None, :]).reshape(K, K * K)
        A = (products.T @ grams.reshape(K, m * m)).reshape(K, K, m, m)
        A = A.transpose(0, 2, 1, 3).reshape(K * m, K * m) / eta
        A.flat[::K * m + 1] += 1
        L = cholesky(A, lower=True, check_finite=False)
        rhs = (R.T @ response).reshape(-1) / eta
        Z = (R[:, :, None] * q[None, None, :]).reshape(K, K * m)
        forward = solve_triangular(L, Z.T, lower=True, check_finite=False)
        mean = Z @ cho_solve((L, True), rhs, check_finite=False)
        variance = np.maximum(np.square(forward).sum(0), 1e-12)
        self.catalog_diagnostics = dict(min_action_eigenvalue=float(eigenvalues.min()),
            clipped_negative_eigenvalues=int((eigenvalues < 0).sum()), action_directions=K,
            kernel_max_abs_error=kernel_error, kernel_error_tolerance=float(tolerance))
        return torch.as_tensor(mean, dtype=self.B.dtype), torch.as_tensor(variance, dtype=self.B.dtype)

    def mean_var(self, theta, X):
        return self._posterior(X, theta)

    def posterior(self, x: np.ndarray, theta: np.ndarray):
        mean, var = self._posterior(np.asarray(x)[None], theta)
        return float(mean[0]), float(var[0])

    def select_action(self, X_candidates: np.ndarray, theta: np.ndarray):
        mean, var = self._posterior(X_candidates, theta)
        beta = self.beta
        if beta is None:
            beta = 2 * math.log(len(X_candidates) * (self.step_count + 1) ** 2 * math.pi ** 2 /
                                (2 * self.c.get("delta", .1)))
        self.last_mean, self.last_var = mean.cpu().numpy(), var.cpu().numpy()
        idx = int(torch.argmax(mean + self.c.get("c", 1.) * math.sqrt(beta) * var.sqrt()))
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

        if calibration_contexts is not None or self.calibration_mode == "auto":
            self.calibration = calibration_contexts
        self.spectral(self.step_count)

    def spectral(self, t):
        if (self.prune_rule == "none" or self._spectral_frozen or
                t < self.burn_in or t % self.T_spec):
            return
        self._update_projection(self.calibration, t=t)
        if self.fold_round is not None and t == self.fold_round:
            self.transfer = self._fold() if self.m < self.m_max else dict(rank=self.m_max, skipped=True)
            self._spectral_frozen = True

    def _update_projection(self, calibration_contexts=None, t=None):
        t = self.step_count if t is None else t
        if self.calibration_mode == "auto":
            self._spectral_backend = "support" if self.prune_rule == "bbp" else "numpy"
        if self._spectral_backend == "torch":
            self._torch_projection(t)
            return
        if self.calibration_mode == "fresh" and calibration_contexts is None:
            raise ValueError("Fresh calibration requires calibration_contexts on spectral rounds")
        contexts = (self.th if self.calibration_mode == "history" or calibration_contexts is None else
                    torch.as_tensor(calibration_contexts, dtype=self.B.dtype, device=self.device))
        if contexts.ndim != 2 or contexts.shape[0] == 0 or contexts.shape[1] != self.theta_dim:
            raise ValueError("Calibration contexts must have shape (n, theta_dim), n > 0")
        with torch.no_grad():
            H = self.features(contexts).cpu().numpy()
            if self._spectral_backend == "numpy":
                self._numpy_projection(H, contexts, t)
            else:
                self._support_projection(H, t)
            if self.calibration_mode == "auto" or self.debug_spectral:
                row = self.spectral_history[-1]
                values = np.asarray(row["eigenvalues"])
                top = values[:20].copy()
                row.update(top_values=top, top_ratios=top / values[0] if len(top) and values[0] > 0 else None)
                if self.debug_spectral:
                    ratios = top[:10] / values[0] if len(top) and values[0] > 0 else np.zeros_like(top[:10])
                    cutoff = row.get("cutoff", row.get("edge"))
                    print(f"[spectral] t={t:>4} n={len(H):>4} "
                          f"d={row.get('dimension', row.get('support_dimension', self.m_max)):>4} "
                          f"m={self.m:>3} eps={self.eps_prune:.4g} cutoff={cutoff} "
                          f"lambda_1={row['lambda1']:.4g} "
                          f"top{len(ratios)}_ratios=[{', '.join(f'{v:.4f}' for v in ratios)}]", flush=True)
                    counts = {eps: int(np.sum(values > eps * row['lambda1'])) for eps in (1e-4, 1e-3, 1e-2, 1e-1)}
                    print(f"           counts at eps*lambda_1: {counts}", flush=True)

    def _set_projection(self, projection, rank):
        self.m = rank
        self.projection = torch.as_tensor(projection.copy(), dtype=self.B.dtype, device=self.device)
        self.w = torch.ones(rank, dtype=self.B.dtype, device=self.device)

    def _numpy_projection(self, H, contexts, t):
        _, singular, vectors = np.linalg.svd(H, full_matrices=False)
        values = singular * singular / len(H)
        epsilon = None if self.decision_rank is not None else self.eps_prune
        k = self.decision_rank or max(1, int((values > epsilon * values[0]).sum()))
        if k > len(vectors):
            raise ValueError("Fixed rank exceeds the available calibration SVD directions")
        self._set_projection(vectors[:k].T, k)
        self.proposed_rank, self.no_peak, self.epsilon = k, False, epsilon
        rule = "fixed" if epsilon is None else "fixed_epsilon" if self.method in (
            "Relative cutoff 1e-5", "Relative cutoff 1e-6") else "relative"
        row = dict(t=t, rule=rule, n_cal=len(H), rank=k, proposed_rank=k, no_peak=False,
                   epsilon=epsilon, lambda1=float(values[0]), eigenvalues=values.tolist())
        if self.calibration_mode == "auto":
            row.update(n=len(H), dimension=self.m_max, cutoff=float(epsilon * values[0]), background=None)
        self.spectral_history.append(row)

    def _support_projection(self, H, t):
        B = self.B.cpu().numpy()
        W, bias = self.encoder.net[-1].weight.cpu().numpy(), self.encoder.net[-1].bias.cpu().numpy()
        support = np.column_stack((B.T @ W, B.T @ bias))
        Q, singular, _ = np.linalg.svd(support, full_matrices=False)
        Q = Q[:, singular > np.finfo(float).eps * max(support.shape) * singular[0]]
        h = H @ Q
        values, vectors = np.linalg.eigh(h.T @ h / len(h))
        values, vectors = np.clip(values[::-1], 0, None), vectors[:, ::-1]
        d, n = len(values), len(h)
        fresh = self.calibration_mode != "history"
        background, edge, bbp_rank = 0., 0., 0
        lambda1 = float(values[0]) if d else 0.
        if d:
            aspect = d / n
            background = (float(np.median(values[:min(d, n)])) /
                          (max(aspect, 1.) * _mp_median(min(aspect, 1 / aspect)))) if fresh else (
                          float(np.median(values[:min(d, n)])) / _mp_median(aspect))
            edge = background * (1 + np.sqrt(aspect)) ** 2
            bbp_rank = int((values > edge).sum())
        invalid = not (np.isfinite(background) and background > np.finfo(float).eps * max(lambda1, 0.))
        if invalid:
            bbp_rank = 0
        epsilon = self.eps_prune if self.prune_rule == "relative" else None
        relative_rank = max(1, int((values > .01 * lambda1).sum()))
        proposed = max(1, int((values > epsilon * lambda1).sum())) if epsilon is not None else bbp_rank
        self.epsilon = (float(edge / lambda1) if lambda1 > 0 else None) if epsilon is None else epsilon
        self.proposed_rank, self.no_peak = proposed, proposed == 0
        if d == 0 and epsilon is not None:
            projection = np.linalg.svd(H, full_matrices=False)[2][:1].T
        elif self.calibration_mode == "fresh":
            V = Q @ vectors
            projection = V[:, :proposed] if proposed else np.eye(self.m_max)
        else:
            projection = Q @ vectors[:, :proposed] if proposed else np.eye(self.m_max)
        self._set_projection(projection, proposed or self.m_max)
        row = dict(t=t, n_cal=n, support_dimension=d, epsilon=self.epsilon,
                   rank=self.m, proposed_rank=proposed, no_peak=self.no_peak,
                   invalid_background=bool(invalid), edge=float(edge), lambda1=lambda1,
                   eigenvalues=values.tolist())
        if fresh:
            Hhist = self.features(self.th).cpu().numpy()
            lh = np.linalg.eigvalsh(Hhist.T @ Hhist / len(Hhist))[::-1]
            mean = h.mean(0)
            row.update(bbp_rank=bbp_rank, relative_rank=relative_rank,
                relative_history_rank=int((lh > .01 * lh[0]).sum()), sigma_background=float(background),
                mean_energy_fraction=float(mean @ mean / max(values.sum(), 1e-300)),
                mean_energy_over_background=float(mean @ mean / max(background, 1e-300)),
                support_relative_error=float(np.linalg.norm(H - h @ Q.T) / max(np.linalg.norm(H), 1e-300)),
                tail_energy_fraction=float(values[self.m:].sum() / max(values.sum(), 1e-300)),
                likelihood_noise=float(self.hyp[2].exp()))
        else:
            row.update(background=background,
                support_relative_error=float(np.linalg.norm(H - h @ Q.T) / max(np.linalg.norm(H), 1e-300)),
                distinct_contexts=int(len(np.unique(self.th.cpu().numpy(), axis=0))))
        if self.calibration_mode == "auto":
            row.update(n=n, dimension=d, cutoff=float(edge), background=background)
        self.spectral_history.append(row)

    def _torch_projection(self, t):
        with torch.no_grad():
            h = self.features(self.th)
            full = h.shape[0] < h.shape[1]
            if self.corrected_svd:
                event = dict(t=t)
                _, singular, vectors = _checked_svd(h, full, event)
                if "backend" in event:
                    self.numerical_fallbacks.append(event)
            else:
                _, singular, vectors = torch.linalg.svd(h, full_matrices=full)
            values = self.B.new_zeros(self.m_max)
            values[:len(singular)] = singular.square() / len(h)
            epsilon = self.c.get("eps", self.eps_prune) if self.method == "Relative cutoff 1e-2" else self.eps_prune
            k = self.decision_rank or max(1, int((values > epsilon * values[0]).sum()))
            self.projection, self.w, self.m = vectors.T[:, :k], self.B.new_ones(k), k
            self.proposed_rank, self.no_peak = k, False
            self.epsilon = None if self.decision_rank else epsilon
            self.spectral_history.append(dict(t=t, lam=values.tolist(), noise_var=float(self.hyp[2].exp()),
                                 variance=float(self.hyp[1].exp()), ell=float(self.hyp[0].exp())))

    def _parameter_counts(self):
        neural = sum(p.numel() for p in self.encoder.net.parameters())
        return dict(nn=neural, total=neural + self.B.numel() + self.hyp.numel())

    def _fold(self):
        with torch.no_grad():
            old_features = self.features(self.th, True).clone()
            query = self.th[-1].cpu().numpy() + .137
            actions = self.x.cpu().numpy()
            before = self.mean_var(query, actions)
            prior_parameters = self._parameter_counts()
            U, R = torch.linalg.qr(self.B @ (self.projection * self.w.sqrt()), mode="reduced")
            k, old_head, old_optimizer = R.shape[0], self.encoder.net[-1], self.optimizer
            with torch.random.fork_rng(devices=[]):
                head = nn.Linear(old_head.in_features, k, dtype=self.B.dtype).to(self.device)
            head.weight.copy_(U.T @ old_head.weight)
            head.bias.copy_(U.T @ old_head.bias)
            self.encoder.net[-1], self.B = head, nn.Parameter(R.contiguous())
            self.m_max = self.m = self.proposed_rank = k
            self.c = {**self.c, "M": k}
            self.projection, self.w = torch.eye(k, dtype=self.B.dtype, device=self.device), self.B.new_ones(k)
            current = [*self.encoder.net.parameters(), self.B, self.hyp]
            self.optimizer = optim.Adam(current, **old_optimizer.defaults)
            for p in current:
                if p in old_optimizer.state:
                    self.optimizer.state[p] = copy.deepcopy(old_optimizer.state[p])
            after = self.mean_var(query, actions)
            new_features = self.features(self.th, True)
            errors = [float((old_features - new_features).abs().max())]
            errors += [float((a - b).abs().max()) for a, b in zip(before, after)]
            torch.testing.assert_close(new_features, old_features, atol=1e-9, rtol=1e-9)
            old_kernel = self.kernel(old_features, old_features, self.D2)
            new_kernel = self.kernel(new_features, new_features, self.D2)
            torch.testing.assert_close(new_kernel, old_kernel, atol=1e-8, rtol=1e-8)
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
            assert {id(p) for p in current} == {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            assert old_head.weight not in self.optimizer.state and old_head.bias not in self.optimizer.state
        return dict(rank=k, before=prior_parameters, after=self._parameter_counts(),
                    feature_error=errors[0], mean_error=errors[1], variance_error=errors[2],
                    kernel_error=float((old_kernel - new_kernel).abs().max()))


    def prior_variance(self, x: np.ndarray, theta: np.ndarray) -> float:
        with torch.no_grad():
            theta_t = torch.as_tensor(theta, dtype=self.B.dtype, device=self.device).unsqueeze(0)
            h_t = self.features(theta_t, True)
            return float((h_t.square().sum() * self.variance).cpu().item())
