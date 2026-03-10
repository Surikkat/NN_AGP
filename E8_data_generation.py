import numpy as np


def make_sparse_transient_reward(theta_dim=5, x_dim=5,
                                 m_true=2, m_extra=12,
                                 seed=0):
    rng = np.random.RandomState(seed)

    W_t_core = np.zeros((theta_dim, m_true))
    W_x_core = np.zeros((x_dim, m_true))
    for i in range(m_true):
        W_t_core[i, i] = 1.0
        W_x_core[i, i] = 1.0

    m_spike = 3
    W_t_spike = rng.randn(theta_dim, m_spike)
    W_x_spike = rng.randn(x_dim, m_spike)

    W_t_noise = rng.randn(theta_dim, m_extra) * 0.8
    W_x_noise = rng.randn(x_dim, m_extra) * 0.8

    W_t = np.concatenate([W_t_core, W_t_spike, W_t_noise], axis=1)
    W_x = np.concatenate([W_x_core, W_x_spike, W_x_noise], axis=1)

    def reward(theta, x, t):
        g = theta @ W_t
        p = x @ W_x
        r = 0.0

        for k in range(m_true):
            z = g[k] * p[k]
            r += 1.2 * np.tanh(0.9 * z)
            r += 0.3 * np.sin(4.0 * z)

        if 70 <= t <= 120:
            for k in range(m_true, m_true + m_spike):
                z = g[k] * p[k]
                r += 0.9 * np.tanh(0.8 * z)

        start = m_true + m_spike
        end = start + m_extra
        for k in range(start, end):
            phase = 0.3 * t + 5.0 * np.sin(0.05 * t + k)
            r += 0.05 * np.sin(g[k] * p[k] + phase)

        return float(r)

    return reward


def generate_sparse_transient_dataset(
        path="E8_dataset.npz",
        SEED=42,
        theta_dim=5,
        x_dim=5,
        T=200,
        m_true=2,
        m_extra=12,
        pool_size=20,
        noise_std=0.001):

    rng = np.random.RandomState(SEED)
    X_pool = rng.randn(pool_size, x_dim).astype(np.float32)

    reward_fn = make_sparse_transient_reward(
        theta_dim, x_dim, m_true, m_extra, seed=SEED + 1
    )

    thetas = np.zeros((T, theta_dim), dtype=np.float32)
    rewards = np.zeros((T, pool_size), dtype=np.float32)
    thetas[0] = rng.randn(theta_dim)

    for t in range(T):
        if t > 0:
            thetas[t] = thetas[t - 1] * 0.95 + rng.randn(theta_dim) * 0.25

        theta_t = thetas[t]
        for i in range(pool_size):
            r = reward_fn(theta_t, X_pool[i], t)
            rewards[t, i] = r + rng.randn() * noise_std

    np.savez(
        path,
        X_pool=X_pool,
        thetas=thetas,
        rewards_clean=rewards,
        theta_dim=theta_dim,
        x_dim=x_dim,
        T=T,
        pool_size=pool_size,
        noise_std=noise_std
    )

    print(f"[E8] датасет сохранён в {path}")


if __name__ == "__main__":
    generate_sparse_transient_dataset()

