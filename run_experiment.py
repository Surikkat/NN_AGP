import numpy as np
import torch
import datetime
from tqdm import trange, tqdm


from bandits.nn_agp import NN_AGP_UCB
from bandits.nn_agp_adaptive import NN_AGP_UCB_Adaptive
from bandits.gp_ucb_multikernel import GP_UCB_KernelFlexible
from bandits.neural_ucb import NeuralUCB
from bandits.nn_ucb import NN_UCB


def load_dataset(path="dataset_adapt_adv_highdim.npz"):
    data = np.load(path, allow_pickle=True)
    return data


def run_single(seed, data):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_pool = data["X_pool"]
    thetas = data["thetas"]
    rewards_clean = data["rewards_clean"]
    T = int(data["T"])
    theta_dim = int(data["theta_dim"])
    noise_std = float(data["noise_std"])

    fixed_m_values = [25, 50, 100]

    agents_fixed = {
        m: NN_AGP_UCB(theta_dim, X_pool.shape[1], m=m, hidden_dim=64, lr=2e-3, mll_steps=40, beta=2.0)
        for m in fixed_m_values
    }

    agent_adapt = NN_AGP_UCB_Adaptive(theta_dim, X_pool.shape[1], m=25, hidden_dim=64,
                                     lr=1e-3, mll_steps=20, beta=2.0, T_spec=10, eps_prune=0.1)

    agent_gpucb_mult = GP_UCB_KernelFlexible(X_pool.shape[1], theta_dim, noise_std, beta=2.0)
    agent_gpucb_adapt = GP_UCB_KernelFlexible(X_pool.shape[1], theta_dim, noise_std, beta=2.0,
                                             kernel_type='adaptive', adaptive_weights={'theta': 0.6, 'x': 0.4})

    agent_neuralucb = NeuralUCB(input_dim=X_pool.shape[1], T=T, lambd=1.0, nu=1.0,
                                delta=0.01, S=1.0, eta=1e-3, J=20, m=64, L=2)

    agent_nnucb = NN_UCB(context_dim=X_pool.shape[1], hidden_width=256, hidden_layers=2,
                         lambda_=1.0, beta=1.0, J=10, lr=1e-3, device=None, seed=seed)

    regrets_fixed = {m: [] for m in fixed_m_values}
    inst_fixed = {m: [] for m in fixed_m_values}
    regrets_adapt, inst_adapt, m_history = [], [], []
    regrets_gp_mult, inst_gp_mult = [], []
    regrets_gp_adapt, inst_gp_adapt = [], []
    regrets_neuralucb, inst_neuralucb = [], []
    regrets_nnucb, inst_nnucb = [], []

    cum_fixed = {m: 0.0 for m in fixed_m_values}
    cum_adapt = cum_gp_mult = cum_gp_adapt = cum_neural = cum_nn = 0.0

    for t in trange(T, desc=f"Seed {seed}", leave=False):
        theta_t = thetas[t]
        rewards_row = rewards_clean[t]
        best = rewards_row.max()

        # Фиксированные
        for m, agent in agents_fixed.items():
            x, _, _ = agent.select_action(X_pool, theta_t)
            idx = np.argwhere((X_pool == x).all(axis=1))[0][0]
            r = best - rewards_row[idx]
            agent.update(theta_t, x, float(rewards_row[idx]))
            cum_fixed[m] += r
            regrets_fixed[m].append(cum_fixed[m])
            inst_fixed[m].append(r)

        # Адаптивный
        x, _, _ = agent_adapt.select_action(X_pool, theta_t)
        idx = np.argwhere((X_pool == x).all(axis=1))[0][0]
        r = best - rewards_row[idx]
        agent_adapt.update(theta_t, x, float(rewards_row[idx]))
        cum_adapt += r
        regrets_adapt.append(cum_adapt)
        inst_adapt.append(r)
        m_history.append(getattr(agent_adapt, "m", None))

        # GP mult
        x, _, _ = agent_gpucb_mult.select_action(X_pool, theta_t)
        idx = np.argwhere((X_pool == x).all(axis=1))[0][0]
        r = best - rewards_row[idx]
        agent_gpucb_mult.update(theta_t, x, float(rewards_row[idx]))
        cum_gp_mult += r
        regrets_gp_mult.append(cum_gp_mult)
        inst_gp_mult.append(r)

        # GP adapt
        x, _, _ = agent_gpucb_adapt.select_action(X_pool, theta_t)
        idx = np.argwhere((X_pool == x).all(axis=1))[0][0]
        r = best - rewards_row[idx]
        agent_gpucb_adapt.update(theta_t, x, float(rewards_row[idx]))
        cum_gp_adapt += r
        regrets_gp_adapt.append(cum_gp_adapt)
        inst_gp_adapt.append(r)

        # NeuralUCB
        x, _, _ = agent_neuralucb.select_action(X_pool)
        idx = np.argwhere((X_pool == x).all(axis=1))[0][0]
        r = best - rewards_row[idx]
        agent_neuralucb.update(None, x, float(rewards_row[idx]))
        cum_neural += r
        regrets_neuralucb.append(cum_neural)
        inst_neuralucb.append(r)

        # NN-UCB
        idx = agent_nnucb.select_action(X_pool)
        r = best - rewards_row[idx]
        agent_nnucb.update(X_pool[idx], idx, float(rewards_row[idx]))
        cum_nn += r
        regrets_nnucb.append(cum_nn)
        inst_nnucb.append(r)

    return locals()


def average_dict_list(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {k: np.mean([d[k] for d in list_of_dicts], axis=0) for k in keys}


def average_list(list_of_lists):
    return np.mean(list_of_lists, axis=0)

if __name__ == "__main__":
    data = load_dataset("/home/surikkat/code/NN_AGP/datasets/E2_dataset.npz")

    seeds = [42, 123, 999]
    runs = []

    for s in tqdm(seeds, desc="Прогресс по всем экспериментам"):
        runs.append(run_single(s, data))

    avg = {}

    avg["regrets_fixed"] = average_dict_list([r["regrets_fixed"] for r in runs])
    avg["inst_fixed"] = average_dict_list([r["inst_fixed"] for r in runs])

    for key in [
        "regrets_adapt","inst_adapt",
        "regrets_gp_mult","inst_gp_mult",
        "regrets_gp_adapt","inst_gp_adapt",
        "regrets_neuralucb","inst_neuralucb",
        "regrets_nnucb","inst_nnucb",
        "m_history"
    ]:
        avg[key] = average_list([r[key] for r in runs])

    avg["T"] = runs[0]["T"]

    out_path = f"./results/avg_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

    np.savez(out_path, **avg)
    print(f"\nУсреднённые результаты сохранены в: {out_path}")

