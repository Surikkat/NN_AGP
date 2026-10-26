"""Exact low-rank Gram arithmetic for NeuralUCB."""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Gram:
    def __init__(self, size, diagonal):
        if diagonal <= 0:
            raise ValueError("Positive Gram diagonal required")
        self.size, self.diagonal = size, float(diagonal)
        self.factors = []

    def add(self, vector, scale):
        if scale <= 0:
            raise ValueError("Positive Gram update required")
        self.factors.append(np.asarray(vector, dtype=float).copy() / math.sqrt(scale))

    def dual(self, eps=0.0):
        diagonal = self.diagonal + eps
        U = np.column_stack(self.factors) if self.factors else np.empty((self.size, 0))
        K = np.eye(U.shape[1]) + (U.T @ U) / diagonal
        return diagonal, U, K

    def logdet(self, eps=0.0):
        diagonal, _, K = self.dual(eps)
        _, logdet = np.linalg.slogdet(K)
        return self.size * math.log(diagonal) + logdet

    def inverse(self, eps=0.0):
        return Inverse(*self.dual(eps))


class Inverse:
    def __init__(self, diagonal, U, K):
        self.diagonal, self.U = diagonal, U
        self.factor = cho_factor(K, lower=True, check_finite=False) if len(K) else None

    def dot(self, vector):
        result = vector / self.diagonal
        if self.factor is not None:
            result = result - self.U @ cho_solve(self.factor, self.U.T @ vector, check_finite=False) / self.diagonal**2
        return result
