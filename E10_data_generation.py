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
    W_t_spike_base = rng.randn(theta_dim, m_spike)
    W_x_spike_base = rng.randn(x_dim, m_spike)

    W_t_noise = rng.randn(theta_dim, m_extra) * 0.6
    W_x_noise = rng.randn(x_dim, m_extra) * 0.6

    W_t = np.concatenate([W_t_core, W_t_spike_base, W_t_noise], axis=1)
    W_x = np.concatenate([W_x_core, W_x_spike_base, W_x_noise], axis=1)

    def rotate_block(W, angle):
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        W_rot = W.copy()
        W_rot[:, :2] = W[:, :2] @ R 
        return W_rot

    def reward(theta, x, t):
        r = 0.0

        drift_angle = 0.03 * t
        W_t_spike = rotate_block(W_t_spike_base, drift_angle)
        W_x_spike = rotate_block(W_x_spike_base, drift_angle)

        W_t_full = np.concatenate([W_t_core, W_t_spike, W_t_noise], axis=1)
        W_x_full = np.concatenate([W_x_core, W_x_spike, W_x_noise], axis=1)

        g = theta @ W_t_full
        p = x @ W_x_full

        core_signal = 0.0
        for k in range(m_true):
            z = g[k] * p[k]
            core_signal += 1.5 * np.tanh(1.2 * z)
            core_signal += 0.4 * np.sin(4.0 * z)

            core_signal += 0.35 * np.maximum(0, z - 0.5)

        r += core_signal

        if 40 <= t <= 170:
            gate = np.sign(core_signal)
            for k in range(m_true, m_true + m_spike):
                z = g[k] * p[k]
                r += gate * (
                    1.0 * np.tanh(1.1 * z) +
                    0.45 * np.sin(6.5 * z)
                )

        start = m_true + m_spike
        end = start + m_extra
        for k in range(start, end):
            phase = 0.25 * t + 4.0 * np.sin(0.04 * t + k)
            weight = 0.05 if t <= 150 else 0.015
            r += weight * np.sin(g[k] * p[k] + phase)

        return float(r)


    return reward


def generate_sparse_transient_dataset(
        path="E10_dataset.npz",
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

    print(f"[E10] датасет сохарнён в {path}")


if __name__ == "__main__":
    generate_sparse_transient_dataset()
