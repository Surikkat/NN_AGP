import numpy as np
import datetime
import os
from tqdm import tqdm
from experiments.run_setting_a import run_single


SETTING_A_KWARGS = dict(
    theta_dim=50,
    x_dim=5,
    m_true=5,
    T=200,
    pool_size=50,
    noise_std=0.05,
    ar_coef=0.95,
    ar_noise=0.3,
)


def average_dict_list(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {k: np.mean([d[k] for d in list_of_dicts], axis=0) for k in keys}


def average_list(list_of_lists):
    return np.mean(list_of_lists, axis=0)


if __name__ == "__main__":
    seeds = [42, 123, 999, 2024, 7]
    runs = []

    for s in tqdm(seeds, desc="Setting A"):
        runs.append(run_single(s, SETTING_A_KWARGS))

    avg = {}
    avg["regrets"] = average_dict_list([r["regrets"] for r in runs])
    avg["m_history"] = average_list([r["m_history"] for r in runs])
    avg["T"] = runs[0]["T"]

    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/setting_a_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out_path, **{f"regrets_{k}": v for k, v in avg["regrets"].items()},
             m_history=avg["m_history"], T=avg["T"])
    print(f"\nSaved: {out_path}")

    print("\n=== Final cumulative regret (mean over seeds) ===")
    for k, v in avg["regrets"].items():
        print(f"{k:>15s}: {v[-1]:.2f}")