# experiments/setting_a_highdim_lowrank.py
import numpy as np


def make_setting_a_reward(theta_dim=100, x_dim=5, m_true=5, seed=0):
    rng = np.random.RandomState(seed)
    W_g = np.zeros((theta_dim, m_true))
    for i in range(m_true):
        W_g[i, i] = 1.0
    W_p = rng.randn(x_dim, m_true)
    Q, _ = np.linalg.qr(W_p)
    W_p = Q
    spike = np.array([1.0, 0.9, 0.8, 0.7, 0.6])[:m_true]

    def reward(theta, x):
        g = theta @ W_g
        p = x @ W_p
        return float(np.sum(spike * g * p))
    return reward, W_g, W_p, spike


def generate_setting_a_on_the_fly(
    theta_dim=100,
    x_dim=5,
    m_true=5,
    T=150,
    pool_size=50,
    noise_std=0.1,
    ar_coef=0.95,
    ar_noise=0.3,
    signal_scale=1.0,
    noise_scale=0.01,
    drift_rate=0.02,          # <-- ЕСТЬ
    burst_range=(80, 120),    # <-- ЕСТЬ
    burst_m=15,               # <-- ЕСТЬ
    seed=42,
):
    rng = np.random.RandomState(seed)
    reward_fn, W_g, W_p, spike = make_setting_a_reward(
        theta_dim, x_dim, m_true, seed=seed + 1
    )

    X_pool = rng.randn(pool_size, x_dim).astype(np.float32)

    theta = np.zeros(theta_dim, dtype=np.float32)
    theta[:m_true] = signal_scale * rng.randn(m_true)
    theta[m_true:] = noise_scale * rng.randn(theta_dim - m_true)

    for t in range(T):
        # ==== Дрейф подпространства ====
        angle = drift_rate * t
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        W_p_t = W_p.copy()
        W_p_t[:, :2] = W_p_t[:, :2] @ R.T

        # ==== Transient burst ====
        m_t = burst_m if burst_range[0] <= t <= burst_range[1] else m_true

        # ==== Генерация theta ====
        if t > 0:
            noise_sig = ar_noise * rng.randn(m_t)
            noise_bg = ar_noise * noise_scale * rng.randn(theta_dim - m_t)
            theta[:m_t] = ar_coef * theta[:m_t] + signal_scale * noise_sig
            theta[m_t:] = ar_coef * theta[m_t:] + noise_bg

        # ==== Rewards ====
        m_eff = min(m_t, m_true)
        rewards = np.array([
            np.sum(
                spike[:m_eff]
                * (theta[:m_eff] @ W_g[:m_eff, :m_eff])
                @ (X_pool[i] @ W_p_t[:, :m_eff])
            )
            for i in range(pool_size)
        ])
        rewards = rewards + rng.randn(pool_size) * noise_std
        yield theta, X_pool, rewards