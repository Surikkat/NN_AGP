import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List

torch.set_default_dtype(torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim: int, m: int, L: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(L):
            layers.append(nn.Linear(in_dim, m))
            layers.append(nn.ReLU())
            in_dim = m
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class NeuralUCB:
    def __init__(self,
                 input_dim: int,
                 T: int,
                 lambd: float,
                 nu: float,
                 delta: float,
                 S: float,
                 eta: float,
                 J: int,
                 m: int,
                 L: int,
                 C1: float = 1.0,
                 C2: float = 1.0,
                 C3: float = 1.0,
                 device: str = "cpu",
                 dual: bool = False):
        self.device = device
        self.input_dim = input_dim
        self.T = T
        self.lambd = lambd
        self.nu = nu
        self.delta = delta
        self.S = S
        self.eta = eta
        self.J = J
        self.m = m
        self.L = L
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.dual = dual

        self.model = MLP(input_dim, m, L).to(self.device)
        self.theta0_vec = self._get_param_vector(self.model).copy()

        self.p = self.theta0_vec.size
        if self.dual:
            from .neural_ucb_dual import Gram
            self.Z = Gram(self.p, lambd)
        else:
            self.Z = (lambd * np.eye(self.p)).astype(np.float64)

        self.history_X = []
        self.history_y = []

        self._Z_inv_cache = None
        self._Z_inv_valid = False

        self._eps = 1e-12

    def _get_param_vector(self, model: nn.Module) -> np.ndarray:
        with torch.no_grad():
            vecs = []
            for p in model.parameters():
                vecs.append(p.view(-1).cpu().numpy())
            if vecs:
                return np.concatenate(vecs).astype(np.float64)
            else:
                return np.zeros(0, dtype=np.float64)

    def _set_param_vector(self, model: nn.Module, vec: np.ndarray):
        vec = vec.astype(np.float32)
        pointer = 0
        with torch.no_grad():
            for p in model.parameters():
                numel = p.numel()
                chunk = vec[pointer:pointer+numel]
                p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)).to(p.device))
                pointer += numel
        assert pointer == vec.size

    def _grad_wrt_params(self, x_np: np.ndarray, use_model: Optional[nn.Module] = None) -> np.ndarray:
        model = use_model if use_model is not None else self.model
        model.zero_grad()
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(self.device).unsqueeze(0)
        out = model(x_t)
        grads = torch.autograd.grad(out, [p for p in model.parameters()], retain_graph=False, create_graph=False)
        vecs = [g.contiguous().view(-1).cpu().numpy().astype(np.float64) for g in grads]
        return np.concatenate(vecs)

    def compute_gamma(self, t: int) -> float:
        t = max(2, int(t))

        C1, C2, C3 = self.C1, self.C2, self.C3
        m, L, lambd, eta, J, nu, delta, S = self.m, self.L, self.lambd, self.eta, self.J, self.nu, self.delta, self.S

        if self.dual:
            logdetZ = self.Z.logdet(self._eps)
            logdetLam = self.p * math.log(lambd)
        else:
            signZ, logdetZ = np.linalg.slogdet(self.Z + self._eps*np.eye(self.p))
            signLam, logdetLam = np.linalg.slogdet(lambd * np.eye(self.p))
        log_det_ratio = logdetZ - logdetLam
        if log_det_ratio < 0:
            log_det_ratio = max(log_det_ratio, -1e6)

        log_term1 = np.log(max(2.0, m * (L**4) * t))
        log_term2 = np.log(max(2.0, m * (L**(7/2)) * t))

        A = C1 * m**(-1.0/6.0) * np.sqrt(log_term1 * (t**(7.0/6.0)) * (lambd**(-7.0/6.0)))

        B = (
            nu * np.sqrt(max(0.0, log_det_ratio))
            + C2 * m**(-1.0/6.0) * np.sqrt(log_term1 * (t**(5.0/3.0)) * (lambd**(-1.0/6.0)))
            - 2.0 * np.log(delta)
            + np.sqrt(lambd) * S
        )

        first_block = np.sqrt(1.0 + A * B)

        decay_term = (1.0 - eta * m * lambd)
        if decay_term < 0:
            decay_term = 0.0

        sgd_term = (lambd + C3 * t * L) * (
            (decay_term**(J/2.0)) * np.sqrt(t / max(1e-12, lambd))
            + m**(-1.0/6.0) * np.sqrt(log_term2 * (t**(5.0/3.0)) * (lambd**(-5.0/3.0))) * (1.0 + np.sqrt(t / max(1e-12, lambd)))
        )

        gamma_t = float(first_block + sgd_term)
        return gamma_t

    def _ensure_Z_inv(self):
        if not self._Z_inv_valid:
            if self.dual:
                self._Z_inv_cache = self.Z.inverse(self._eps)
            else:
                try:
                    self._Z_inv_cache = np.linalg.inv(self.Z + self._eps*np.eye(self.p))
                except np.linalg.LinAlgError:
                    self._Z_inv_cache = np.linalg.pinv(self.Z + self._eps*np.eye(self.p))
            self._Z_inv_valid = True
        return self._Z_inv_cache

    def select_action(self, X_pool: np.ndarray, theta_vec: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        if theta_vec is not None:
            self._set_param_vector(self.model, theta_vec)

        t = max(1, len(self.history_y) + 1)
        gamma_t = self.compute_gamma(t)

        Z_inv = self._ensure_Z_inv()

        utilities = []
        K = X_pool.shape[0]
        for i in range(K):
            x = X_pool[i]
            with torch.no_grad():
                xt = torch.from_numpy(x.astype(np.float32)).to(self.device).unsqueeze(0)
                f_val = float(self.model(xt).cpu().numpy().squeeze())

            g_vec = self._grad_wrt_params(x)
            q = float(g_vec.dot(Z_inv.dot(g_vec)))
            u = f_val + gamma_t * np.sqrt(max(0.0, q) / float(self.m))
            utilities.append(float(u))

        utilities = list(utilities)
        idx = int(np.argmax(utilities))
        x_sel = X_pool[idx]
        return x_sel, utilities, idx

    def update(self, theta_vec: Optional[np.ndarray], x_chosen: np.ndarray, reward: float):
        if theta_vec is not None:
            self._set_param_vector(self.model, theta_vec)

        self.history_X.append(x_chosen.astype(np.float32))
        self.history_y.append(float(reward))

        g_vec = self._grad_wrt_params(x_chosen)
        if self.dual:
            self.Z.add(g_vec, float(self.m))
        else:
            self.Z += np.outer(g_vec, g_vec) / float(self.m)
        self._Z_inv_valid = False
        self._train_nn()

    def _train_nn(self):
        if len(self.history_y) == 0:
            return

        optimizer = optim.SGD(self.model.parameters(), lr=self.eta, momentum=0.0)
        X = torch.from_numpy(np.stack(self.history_X).astype(np.float32)).to(self.device)
        y = torch.from_numpy(np.array(self.history_y, dtype=np.float32)).to(self.device)

        theta0 = torch.from_numpy(self.theta0_vec.astype(np.float32)).to(self.device)

        for _ in range(self.J):
            optimizer.zero_grad()
            preds = self.model(X)
            mse = 0.5 * torch.mean((preds - y)**2) * X.shape[0]
            cur_vec = torch.cat([p.view(-1) for p in self.model.parameters()])
            reg = 0.5 * float(self.m * self.lambd) * torch.sum((cur_vec - theta0)**2)
            loss = mse + reg
            loss.backward()
            optimizer.step()
        return

    def get_theta_vector(self) -> np.ndarray:
        return self._get_param_vector(self.model)

    def reset(self):
        self.history_X = []
        self.history_y = []
        if self.dual:
            from .neural_ucb_dual import Gram
            self.Z = Gram(self.p, self.lambd)
        else:
            self.Z = (self.lambd * np.eye(self.p)).astype(np.float64)
        self._Z_inv_valid = False
        self._set_param_vector(self.model, self.theta0_vec.copy())

    
