# experiments/run_one_setting_a.py
import numpy as np
import os
import matplotlib.pyplot as plt

from experiments.run_setting_a import run_single

SETTING_A_KWARGS = dict(
    theta_dim=100, x_dim=5, m_true=5,
    T=150, pool_size=50, noise_std=0.1,
    ar_coef=0.95, ar_noise=0.3,
    drift_rate=0.02,
    burst_range=(80, 120),
    burst_m=15,
)

if __name__ == "__main__":
    seed = 42
    result = run_single(
        seed, SETTING_A_KWARGS,
        log_every=10, verbose=True, debug_spectral=True,
    )
    print(f"\nElapsed: {result['elapsed']:.1f}s")
    print(f"\nm_fixed (formula) = {result['m_fixed']}, m_true = {result['m_true']}")

    print("\n=== m_history (adapt_bbp) ===")
    mh = np.array(result["m_history"]["adapt_bbp"], dtype=float)
    print(f"first 10: {mh[:10]}")
    print(f"last 10 : {mh[-10:]}")
    print(f"median  : {np.nanmedian(mh):.2f}")
    print(f"min/max : {np.nanmin(mh):.0f} / {np.nanmax(mh):.0f}")

    os.makedirs("./results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for k, v in result["regrets"].items():
        ax.plot(v, label=k)
    ax.axvspan(80, 120, alpha=0.1, color="red", label="burst")
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"Setting A (sparse + drift + burst), seed={seed}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result["m_history"]["adapt_bbp"], label="m (adapt_bbp)", color="tab:red")
    ax.axhline(y=result["m_fixed"], linestyle="--", color="black",
               alpha=0.5, label=f"m_fixed = {result['m_fixed']}")
    ax.axhline(y=result["m_true"], linestyle=":", color="gray",
               alpha=0.5, label=f"m* = {result['m_true']}")
    ax.axvspan(80, 120, alpha=0.1, color="red")
    ax.set_xlabel("round")
    ax.set_ylabel("selected dimension")
    ax.set_title("Adaptive m (BBP) over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"./results/setting_a_sparse_seed{seed}.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot: {out_path}")