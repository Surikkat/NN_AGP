# experiments/run_setting_a.py
import numpy as np
import torch
import math
import time

from bandits.nn_agp import NN_AGP_UCB
from bandits.nn_agp_adaptive import NN_AGP_UCB_Adaptive
from bandits.gp_ucb_multikernel import GP_UCB_KernelFlexible
from bandits.neural_ucb import NeuralUCB
from bandits.nn_ucb import NN_UCB

from experiments.setting_a_highdim_lowrank import generate_setting_a_on_the_fly


def recommended_m(theta_dim: int, x_dim: int) -> int:
    return math.ceil(theta_dim / 3 + x_dim / 10) + 3


def run_single(seed, setting_kwargs, log_every=10, verbose=True, debug_spectral=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    theta_dim = setting_kwargs["theta_dim"]
    x_dim = setting_kwargs["x_dim"]
    T = setting_kwargs["T"]
    noise_std = setting_kwargs["noise_std"]
    m_true = setting_kwargs.get("m_true", None)
    m_fixed = recommended_m(theta_dim, x_dim)

    COMMON_LR = 1e-3
    COMMON_MLL_STEPS = 40
    COMMON_BETA = 2.0
    COMMON_INIT_NOISE = noise_std

    if verbose:
        print(f"=== Single run, Setting A, seed={seed} ===")
        print(f"theta_dim={theta_dim}, x_dim={x_dim}, T={T}, noise_std={noise_std}")
        print(f"m_fixed (formula) = {m_fixed}, m_true = {m_true}")
        print(f"common: lr={COMMON_LR}, mll_steps={COMMON_MLL_STEPS}, "
              f"beta={COMMON_BETA}, init_noise={COMMON_INIT_NOISE}")
        print()

    # ==== 1. Fixed NN-AGP (рекомендация статьи) ====
    agent_fixed = NN_AGP_UCB(
        theta_dim, x_dim, m=m_fixed, hidden_dim=64,
        lr=COMMON_LR, mll_steps=COMMON_MLL_STEPS,
        beta=COMMON_BETA, init_noise=COMMON_INIT_NOISE,
    )

    # ==== 2. Adaptive, burn_in=20, BBP (наш метод) ====
    agent_adapt_bbp = NN_AGP_UCB_Adaptive(
        theta_dim, x_dim, m=m_fixed, hidden_dim=64,
        lr=COMMON_LR, mll_steps=COMMON_MLL_STEPS,
        beta=COMMON_BETA, init_noise=COMMON_INIT_NOISE,
        T_spec=10, eps_prune=0.001, burn_in=20,
        prune_rule="bbp",
        debug_spectral=debug_spectral,
    )

    # ==== Baselines ====
    agent_gp_mult = GP_UCB_KernelFlexible(
        x_dim, theta_dim, noise_std, beta=2.0, kernel_type='multiplicative'
    )
    agent_gp_adapt = GP_UCB_KernelFlexible(
        x_dim, theta_dim, noise_std, beta=2.0,
        kernel_type='adaptive', adaptive_weights={'theta': 0.6, 'x': 0.4}
    )
    agent_neuralucb = NeuralUCB(
        input_dim=theta_dim + x_dim, T=T, lambd=1.0, nu=1.0,
        delta=0.01, S=1.0, eta=1e-3, J=20, m=64, L=2,
    )
    agent_nnucb = NN_UCB(
        context_dim=theta_dim + x_dim, hidden_width=256, hidden_layers=2,
        lambda_=1.0, beta=1.0, J=10, lr=1e-3, seed=seed,
    )

    # ==== Колонки ====
    keys = ["fixed", "adapt_bbp", "gp_mult", "gp_adapt", "neuralucb", "nnucb"]
    regrets = {k: [] for k in keys}
    m_history = {"adapt_bbp": []}
    cum = {k: 0.0 for k in keys}

    if verbose and log_every:
        header = f"{'t':>4} | " + " | ".join(f"{k:>10}" for k in keys) + \
                 f" | {'m_bbp':>5} | {'best_r':>8}"
        print(header)
        print("-" * len(header))

    t_start = time.time()

    for t, (theta, X_pool, rewards) in enumerate(
        generate_setting_a_on_the_fly(seed=seed, **setting_kwargs)
    ):
        best = rewards.max()

        # fixed
        x, _, _ = agent_fixed.select_action(X_pool, theta)
        idx = int(np.argmin(np.linalg.norm(X_pool - x, axis=1)))
        r = best - rewards[idx]
        agent_fixed.update(theta, x, float(rewards[idx]))
        cum["fixed"] += r
        regrets["fixed"].append(cum["fixed"])

        # adapt_bbp
        x, _, _ = agent_adapt_bbp.select_action(X_pool, theta)
        idx = int(np.argmin(np.linalg.norm(X_pool - x, axis=1)))
        r = best - rewards[idx]
        agent_adapt_bbp.update(theta, x, float(rewards[idx]))
        cum["adapt_bbp"] += r
        regrets["adapt_bbp"].append(cum["adapt_bbp"])
        m_history["adapt_bbp"].append(getattr(agent_adapt_bbp, "m", None))

        # gp_mult
        x_gp, _, _ = agent_gp_mult.select_action(X_pool, theta)
        idx = int(np.argmin(np.linalg.norm(X_pool - x_gp, axis=1)))
        r = best - rewards[idx]
        agent_gp_mult.update(theta, X_pool[idx], float(rewards[idx]))
        cum["gp_mult"] += r
        regrets["gp_mult"].append(cum["gp_mult"])

        # gp_adapt
        x_gp2, _, _ = agent_gp_adapt.select_action(X_pool, theta)
        idx = int(np.argmin(np.linalg.norm(X_pool - x_gp2, axis=1)))
        r = best - rewards[idx]
        agent_gp_adapt.update(theta, X_pool[idx], float(rewards[idx]))
        cum["gp_adapt"] += r
        regrets["gp_adapt"].append(cum["gp_adapt"])

        # neuralucb / nnucb
        inputs = np.hstack([np.tile(theta, (X_pool.shape[0], 1)), X_pool])

        x_n, _, idx_n = agent_neuralucb.select_action(inputs)
        r = best - rewards[idx_n]
        agent_neuralucb.update(None, inputs[idx_n], float(rewards[idx_n]))
        cum["neuralucb"] += r
        regrets["neuralucb"].append(cum["neuralucb"])

        idx_nn = agent_nnucb.select_action(inputs)
        r = best - rewards[idx_nn]
        agent_nnucb.update(inputs[idx_nn], idx_nn, float(rewards[idx_nn]))
        cum["nnucb"] += r
        regrets["nnucb"].append(cum["nnucb"])

        if verbose and log_every and (t + 1) % log_every == 0:
            row = f"{t+1:>4} | " + " | ".join(f"{cum[k]:>10.2f}" for k in keys) + \
                  f" | {m_history['adapt_bbp'][-1]:>5} | {best:>8.3f}"
            print(row, flush=True)

    if verbose:
        print("\n=== Final cumulative regret ===")
        for k, v in sorted(regrets.items(), key=lambda kv: kv[1][-1]):
            print(f"{k:>15s}: {v[-1]:8.2f}")

    return {
        "regrets": regrets,
        "m_history": m_history,
        "T": T,
        "elapsed": time.time() - t_start,
        "m_fixed": m_fixed,
        "m_true": m_true,
    }