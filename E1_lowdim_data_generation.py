import numpy as np

def make_adaptive_advantage_reward(theta_dim=5, x_dim=5,
                                   m_true=2, m_extra=3,
                                   scale=0.5, seed=0):
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

    def reward(theta, x):
        g = theta @ W_t
        p = x @ W_x
        core = np.tanh(scale * np.dot(g, p))
        weak = 2 * np.sin(3 * np.dot(g, p))
        return float(core + weak)

    return reward


def generate_adaptive_advantage_dataset(path="E1_lowdim.npz",
                                        SEED=42,
                                        theta_dim=5,
                                        x_dim=5,
                                        T=200,
                                        m_true=2,
                                        m_extra=3,
                                        scale=0.5,
                                        noise_std=0.02,
                                        pool_size=50):
    rng = np.random.RandomState(SEED)

    X_pool = rng.randn(pool_size, x_dim).astype(np.float32)
    true_reward_fn = make_adaptive_advantage_reward(
        theta_dim, x_dim, m_true, m_extra, scale, seed=SEED + 1
    )

    thetas = np.zeros((T, theta_dim), dtype=np.float32)
    rewards_clean = np.zeros((T, pool_size))

    thetas[0] = rng.randn(theta_dim)

    for t in range(T):

        if t > 0:
            thetas[t] = thetas[t-1] * 0.95 + rng.randn(theta_dim) * 0.3

        theta_t = thetas[t]

        for i in range(pool_size):
            r = true_reward_fn(theta_t, X_pool[i])
            rewards_clean[t, i] = r + rng.randn() * noise_std

    np.savez(path,
             X_pool=X_pool,
             thetas=thetas,
             rewards_clean=rewards_clean,
             theta_dim=theta_dim,
             x_dim=x_dim,
             T=T,
             pool_size=pool_size,
             noise_std=noise_std)

    print(f"E1 датасет сохранён в {path}")


if __name__ == "__main__":
    generate_adaptive_advantage_dataset()



