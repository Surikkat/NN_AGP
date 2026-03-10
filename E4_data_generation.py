import numpy as np


def make_adaptive_advantage_reward(theta_dim=5, x_dim=5,
                                   m_true=2, m_extra=3,
                                   seed=0):
    rng = np.random.RandomState(seed)

    W_t_true = np.zeros((theta_dim, m_true))
    W_x_true = np.zeros((x_dim, m_true))
    for i in range(m_true):
        W_t_true[i, i] = 1.0
        W_x_true[i, i] = 1.0

    W_t_extra = rng.randn(theta_dim, m_extra)
    W_x_extra = rng.randn(x_dim, m_extra)

    W_t = np.concatenate([W_t_true, W_t_extra], axis=1)
    W_x = np.concatenate([W_x_true, W_x_extra], axis=1)

    def reward(theta, x, t):
        g = theta @ W_t
        p = x @ W_x

        if t < 50:
            active = 6
        elif t < 120:
            active = 4
        else:
            active = 3

        r = 0.0

        for k in range(active):
            scale = 1.0 if k < 3 else 0.7
            r += scale * np.tanh(0.8 * g[k] * p[k])
            r += scale * 0.3 * np.sin(4.0 * g[k] * p[k])

        for k in range(active, m_true + m_extra):
            r += 0.005 * g[k] * p[k] * rng.randn()

        return float(r)

    return reward



def generate_adaptive_advantage_dataset(
        path="E4_dataset.npz",
        SEED=42,
        theta_dim=5,
        x_dim=5,
        T=200,
        m_true=2,
        m_extra=10,
        scale=0.5,
        noise_std=0.005,
        pool_size=20):

    rng = np.random.RandomState(SEED)

    X_pool = rng.randn(pool_size, x_dim).astype(np.float32)
    true_reward_fn = make_adaptive_advantage_reward(
        theta_dim, x_dim, m_true, m_extra, seed=SEED + 1
    )

    thetas = np.zeros((T, theta_dim), dtype=np.float32)
    rewards_clean = np.zeros((T, pool_size), dtype=np.float32)

    thetas[0] = rng.randn(theta_dim)

    for t in range(T):
        if t > 0:
            thetas[t] = thetas[t - 1] * 0.95 + rng.randn(theta_dim) * 0.3

        theta_t = thetas[t]
        for i in range(pool_size):
            r = true_reward_fn(theta_t, X_pool[i], t)
            rewards_clean[t, i] = r + rng.randn() * noise_std

    np.savez(
        path,
        X_pool=X_pool,
        thetas=thetas,
        rewards_clean=rewards_clean,
        theta_dim=theta_dim,
        x_dim=x_dim,
        T=T,
        pool_size=pool_size,
        noise_std=noise_std
    )

    print(f"[E4] датасет сохранён в {path}")


if __name__ == "__main__":
    generate_adaptive_advantage_dataset()

