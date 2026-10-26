"""Build paper tables and figures."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent / "paper_experiments"
ONLINE = ["online.weak", "online.balanced", "online.kuairec600", "online.rank12",
          "online.rank3", "online.rank3_d128"] + [f"v4.exp{i:02d}" for i in range(1, 11)]
NAMES = dict(zip(ONLINE, ["Weak", "Balanced", "KuaiRec", "Rank 12", "Rank 3",
             "Rank 3, 128 inputs"] + [f"Exp. {i}" for i in range(1, 11)]))
TITLES = ["Weak", "Balanced", "KuaiRec 100 users (T=600)", "Rank 12 (noise 0.5)",
          "Rank 3 (noise 0.5)", "Rank 3, 128 inputs (noise 0.5)",
          "Exp 1: Dominant factors", "Exp 2: Rank 3", "Exp 3: Rank 10 with weak factors",
          "Exp 4: Nonlinear rewards", "Exp 5: Heavy-tailed noise", "Exp 6: Growing rank",
          "Exp 7: Near-cutoff nuisance", "Exp 8: Rank spike", "Exp 9: Gradual rotation",
          "Exp 10: Abrupt rotation"]
COLOR = dict(Full="#555555", BBP="#0072B2", Relative="#CC79A7", Small="#D55E00",
             FixedEps1e6="#009E73", Eps1e5="#999999", Fixed10="#7A5195", Fixed25="#A0782C")
LABEL = dict(Full="Full", BBP="BBP", Relative=r"$\varepsilon_{\mathrm{prune}}=10^{-2}$",
             Small=r"$\varepsilon_{\mathrm{prune}}=10^{-4}$",
             Eps1e5=r"$\varepsilon_{\mathrm{prune}}=10^{-5}$",
             FixedEps1e6=r"$\varepsilon_{\mathrm{prune}}=10^{-6}$",
             Fixed10="Rank 10", Fixed25="Rank 25")
STYLE = {"BBP": "-", "Small": "--", "FixedEps1e6": "-.", "Fixed10": ":",
         "Fixed25": (0, (5, 2)), "Relative": (0, (1, 1)), "Eps1e5": (0, (5, 1, 1, 1))}
BASELINES = ("Fixed", "GP mult", "GP add", "NeuralUCB", "NN-UCB")
BASELINE_LABELS = ("Fixed NN-AGP ($m=7$)", "GP-UCB multiplicative",
                   "GP-UCB weighted-additive", "NeuralUCB", "NN-UCB")


@lru_cache(maxsize=128)
def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_path(directory, plan, method, seed):
    return directory / plan["id"] / f"seed{seed}_{method.replace(' ', '_')}" / "result.json"


def load(plans, directory):
    import numpy as np
    experiments = {}
    for plan in plans:
        if plan.get("kind") == "spectral_diagnostic":
            continue
        by_method = {}
        for method in plan["methods"]:
            records = {r["seed"]: r for r in plan["runs"] if r["method"] == method}
            rows = []
            for seed in plan["paper_seed_order"]:
                if seed not in records:
                    continue
                record = records[seed]
                path = ROOT / record["result_file"] if directory is None else result_path(directory, plan, method, seed)
                raw = read(path)
                if raw.get("status") != "complete" or raw["seed"] != seed:
                    raise ValueError(f"Incomplete result: {path}")
                if directory is not None:
                    verify_run(raw, plan, method, seed, path)
                if directory is None and sha(path) != record["source_sha256"]:
                    raise ValueError(f"Saved result changed: {path}")
                if "r" in raw:
                    regret, rank = raw["r"], raw["mhat"]
                    epsilon = [None] * len(regret)
                    width = [plan["configurations"][record["configuration"]]["agent"]["M"]] * len(regret)
                    final_rank = raw.get("mhat_after", rank)[-1]
                else:
                    sequence = raw["trajectory"]
                    if "method" not in raw:
                        sequence = [row[record["raw_method"]] for row in sequence]
                    regret = [row["regret"] for row in sequence]
                    rank = [row.get("rank") for row in sequence]
                    epsilon = [row.get("epsilon") for row in sequence]
                    width = [row.get("nn_width") for row in sequence]
                    final_rank = raw.get("summary", {}).get("final_rank", rank[-1])
                if len(regret) != plan["horizon"] or not np.isfinite(regret).all():
                    raise ValueError(f"Invalid trajectory: {path}")
                regret = np.cumsum(regret)
                if directory is None and not np.isclose(regret[-1], record["regret"], rtol=1e-12, atol=1e-8):
                    raise ValueError(f"Regret sum differs: {path}")
                timing = raw.get("times", {}).get(record["raw_method"], raw.get("timing", {}))
                parameters = raw.get("parameters", {}).get("final") or {}
                rss = raw.get("peak_rss_bytes", raw.get("memory", {}).get("peak_rss_bytes"))
                rows.append(dict(seed=seed, regret=regret, rank=rank, epsilon=epsilon, width=width,
                    final_rank=final_rank, wall=timing.get("wall", raw.get("wall")),
                    cpu=timing.get("cpu", raw.get("sec")), rss=rss,
                    parameters=parameters.get("total"), neural=parameters.get("neural", parameters.get("nn"))))
            by_method[method] = {key: np.asarray([r[key] for r in rows], dtype=float)
                                 for key in rows[0] if key != "seed"}
            by_method[method]["seeds"] = [r["seed"] for r in rows]
        experiments[plan["paper_key"]] = dict(plan=plan, data=by_method)
    read.cache_clear()
    return experiments


def paired(e, method, baseline="Full", metric="regret", post=0):
    import numpy as np
    a, b = e["data"][method], e["data"][baseline]
    seeds = [s for s in e["plan"]["paper_seed_order"] if s in a["seeds"] and s in b["seeds"]]
    def values(data):
        v = data[metric][[data["seeds"].index(s) for s in seeds]]
        return v[:, -1] - (v[:, post - 1] if post else 0) if metric == "regret" else v
    if not seeds:
        raise ValueError("No paired seeds")
    return values(a), values(b)


def verify_run(raw, plan, method, seed, path):
    if (raw.get("status") != "complete" or raw.get("experiment") != plan["id"] or
        raw.get("method") != method or raw.get("seed") != seed):
        raise ValueError(f"Wrong or incomplete run: {path}")
    if any(v is False for k, v in raw.get("reference", {}).items() if k.startswith("matching_")):
        raise ValueError(f"Run differs from saved results: {path}")


def bootstrap(a, b):
    import numpy as np
    ix = np.random.default_rng(20261015).integers(len(a), size=(5000, len(a)))
    lo, hi = np.quantile(a[ix].mean(1) / b[ix].mean(1), [.025, .975])
    return [float(a.mean() / b.mean()), float(lo), float(hi)]


def interval(values, geometric=False):
    import numpy as np
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    if geometric:
        x = np.log(x)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("A paired interval requires at least two finite observations")
    center = x.mean()
    half = t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    result = np.array([center, center - half, center + half])
    return np.exp(result) if geometric else result


def table(label, caption, columns, header, rows):
    return ("\n".join([r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + columns + "}", r"\toprule", header + r" \\",
        r"\midrule", *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n")


def tables(experiments, output):
    import numpy as np
    summary, content = {}, []
    def add(*args):
        content.append(table(*args))
    physical = [experiments["physical_fold300." + k] for k in ("weak", "balanced", "kuairec")]
    physical_stats = {}
    for name, e in zip(("Weak", "Balanced", "KuaiRec"), physical):
        physical_stats[name] = {metric: interval(a / b, True).tolist()
            for metric in ("parameters", "wall", "cpu", "regret", "rss")
            for a, b in [paired(e, "Late", "BBP", metric)]}
    add("tab:network-cost", "Output-layer reduction relative to BBP with the full encoder. Geometric mean ratios.",
        "@{}lrrr@{}", r"Task & Parameters & Wall time & Regret",
        [[name, *[f"{physical_stats[name][m][0]:.3f}" for m in ("parameters", "wall", "regret")]]
         for name in physical_stats])
    descriptions = ["Five dominant factors and 95 weak components", "Rank 3 with separated eigenvalues",
        "Rank 10 with 40 weak components", r"$\tanh$ of a rank-8 bilinear core",
        "Sparse rank-5 factors and heavy-tailed noise", "Rank grows from 3 to 8",
        "Nuisance directions near the cutoff", r"Transient rank burst $5\to15\to5$",
        "Slowly rotating rank-5 subspace", "Abrupt rank-5 subspace rotation"]
    add("tab:environments-direct", "Synthetic environments. The nominal latent width is 100.",
        "@{}cp{10.4cm}@{}", "Exp & Description", list(enumerate(descriptions, 1)))
    settings = []
    for key in ONLINE[:6]:
        e = experiments[key]
        cfg = next(c for c in e["plan"]["configurations"].values() if "env" in c)
        dims = cfg["env"]
        settings.append([NAMES[key], f"$({dims['d_theta']},{dims['d_x']})$",
            dims.get("n_actions", 16 if key in ONLINE[:2] else 200),
            e["plan"]["horizon"], "Fresh" if key in ONLINE[:2] else "History"])
    settings.append(["Exp. 1--10", "$(10,5)$", 20, 200, "History"])
    add("tab:online-protocol", "Online evaluation settings.", "@{}lrrrl@{}",
        "Task & Inputs (context, action) & Actions & Rounds & Calibration", settings)
    main, controls = [], []
    for key in ONLINE:
        e = experiments[key]
        a, b = paired(e, "BBP")
        ratio = bootstrap(a, b)
        rank = float(e["data"]["BBP"]["rank"][:, 100:].mean())
        summary[key] = dict(n=len(a), full=float(b.mean()), bbp=float(a.mean()), dimension=rank,
                            bbp_full=ratio, controls={})
        main.append([NAMES[key], len(a), f"{b.mean():.1f}", f"{a.mean():.1f}", f"{rank:.1f}",
            f"{ratio[0]:.3f} [{ratio[1]:.3f}, {ratio[2]:.3f}]"])
        row = [NAMES[key]]
        for method in ("Full", "Fixed10", "Fixed25", "Relative", "Small", "Eps1e5", "FixedEps1e6"):
            if method not in e["data"]:
                row.append("--")
            else:
                aa, bb = paired(e, "BBP", method)
                change = round(100 * (aa.mean() / bb.mean() - 1), 2)
                row.append("0.00" if change == 0 else f"{change:+.2f}")
                summary[key]["controls"][method] = bootstrap(aa, bb)
        controls.append(row)
    add("tab:bbp-results", "BBP versus Full on paired seeds. Ratios of mean regret with 95\\% paired bootstrap intervals.",
        "@{}lrrrrl@{}", "Task & Seeds & Full & BBP & GP dimension & BBP/Full", main)
    add("tab:bbp-controls", "Change in mean cumulative regret of BBP relative to each control, in percent.",
        "@{}lrrrrrrr@{}", r"Task & Full & Rank 10 & Rank 25 & $10^{-2}$ & $10^{-4}$ & $10^{-5}$ & $10^{-6}$", controls)
    detail = []
    for metric, label in (("parameters", "Parameters"), ("wall", "Wall time"),
                          ("cpu", "CPU time"), ("regret", "Regret"), ("rss", "Peak RSS")):
        detail.append([label, "Ratio", *[f"{v[metric][0]:.3f}" for v in physical_stats.values()]])
        detail.append(["", r"95\% interval", *[f"[{v[metric][1]:.3f}, {v[metric][2]:.3f}]"
                                                for v in physical_stats.values()]])
    add("tab:network-cost-detail", "Output-layer reduction relative to BBP with the full encoder. Geometric mean ratios and 95\\% paired t intervals in log space.",
        "@{}llrrr@{}", "Quantity & & Weak & Balanced & KuaiRec", detail)
    relative = []
    fixed = {k: [] for k in (3, 5, 10)}
    for i in range(1, 11):
        e = experiments[f"legacy.exp{i:02d}.relative_fixed"]
        a, b = paired(e, "Relative")
        v = interval(a / b, True)
        dimension = float(np.median(e["data"]["Relative"]["rank"][:, -1]))
        relative.append([i, f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]", f"{dimension:g}"])
        for k in fixed:
            a, b = paired(e, "Relative", f"Fixed{k}", post=100)
            fixed[k].append(np.log(a / b))
    add("tab:direct-control", "Relative-cutoff NN--AGP versus Full. Geometric paired regret ratios and 95\\% t intervals.",
        "crr", "Exp & Regret / Full & GP dimension", relative)
    fixed_rows = []
    for k, values in fixed.items():
        v = interval(np.exp(np.asarray(values).mean(0)), True)
        fixed_rows.append([k, f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"])
    add("tab:fixed-rank", "Relative cutoff versus prescribed GP dimensions over rounds 101--200. Paired log ratios averaged equally over ten environments.",
        "cr", "Fixed rank & Regret / Fixed", fixed_rows)
    kuai = [experiments[f"legacy.kuairec{k}.T2000"] for k in (10, 100)]
    kuai_rows = [["Seeds", *[len(e["data"]["Full"]["seeds"]) for e in kuai]]]
    for method, label in (("Full", "Full"), ("Relative", r"$\varepsilon_{\mathrm{prune}}=10^{-2}$")):
        row = [label]
        for e in kuai:
            v = interval(e["data"][method]["regret"][:, -1])
            row.append(r"\shortstack{" + f"{v[0]:.1f}" + r"\\ {[" + f"{v[1]:.1f}, {v[2]:.1f}" + "]}}")
        kuai_rows.append(row)
    kuai_rows.append(["GP dimension", *[f"{np.median(e['data']['Relative']['final_rank']):g}" for e in kuai]])
    add("tab:kuairec", r"Relative-cutoff KuaiRec simulations at $T=2000$. Mean cumulative regret with 95\% t intervals.",
        "@{}lcc@{}", " & 10 users & 100 users", kuai_rows)
    (output / "tables.tex").write_text("\n".join(content))
    ranks = [summary[key]["dimension"] for key in ONLINE[:6]]
    reductions = [100 * (1 - summary[key]["bbp_full"][0]) for key in ONLINE[:6]]
    macros = dict(RankRange=f"{min(ranks):.0f}--{max(ranks):.0f}",
                  RegretReductionRange=f"{min(reductions):.1f}--{max(reductions):.1f}",
                  WeakSeeds=summary[ONLINE[0]]["n"], BalancedSeeds=summary[ONLINE[1]]["n"],
                  KuaiSeeds=summary[ONLINE[2]]["n"])
    (output / "results.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                              for k, v in macros.items()))
    return dict(online=summary, physical=physical_stats, macros=macros)


def figures(experiments, phase, output, summary):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import norm, t
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white"})
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    def save(name, fig):
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=.035,
                    metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(folder / f"{name}.png", bbox_inches="tight", pad_inches=.035, dpi=160)
        plt.close(fig)
    selected = [experiments[key] for key in ONLINE[:3]]
    fig = plt.figure(figsize=(7, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    left = fig.add_subplot(grid[0])
    rank_grid = grid[1].subgridspec(2, 1, height_ratios=[.16, 1], hspace=.10)
    top = fig.add_subplot(rank_grid[0])
    bottom = fig.add_subplot(rank_grid[1], sharex=top)
    methods = ["Small", "FixedEps1e6", "Fixed10", "Fixed25", "BBP"]
    for j, e in enumerate(selected):
        for i, method in enumerate(methods):
            if method not in e["data"]:
                continue
            mid, low, high = bootstrap(*paired(e, method))
            y = j + (i - 2) * .12
            left.plot([low, high], [y, y], color=COLOR[method], lw=1.1)
            left.plot(mid, y, "o", color=COLOR[method], ms=4.2 if method == "BBP" else 2.8)
    left.axvline(1, color=".4", ls="--", lw=.7)
    left.set(yticks=range(3), yticklabels=["Weak", "Balanced", "KuaiRec"], ylim=(2.5, -.5),
             title="(a) Regret relative to Full", xlabel=r"$\bar R_T/\bar R_T^{\mathrm{Full}}$")
    left.grid(axis="x", alpha=.18)
    for e, color, style, name in zip(selected, ["#0072B2", "#D55E00", "#009E73"],
                                    ["-", "--", "-."], ["Weak", "Balanced", "KuaiRec"]):
        q = np.quantile(e["data"]["BBP"]["rank"], [.25, .5, .75], axis=0)
        x = np.arange(1, e["plan"]["horizon"] + 1)
        for ax in (top, bottom):
            ax.step(x, q[1], where="post", color=color, ls=style, lw=1.4, label=name)
            ax.fill_between(x, q[0], q[2], step="post", color=color, alpha=.11, lw=0)
    top.set(ylim=(99.3, 100.7), yticks=[100], title="(b) Selected GP dimension")
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    bottom.set(ylim=(15, 40), xlabel="Round", ylabel="GP coordinates", xlim=(1, 800),
               xticks=[100, 400, 800], yticks=[15, 20, 25, 30, 35, 40])
    bottom.legend(frameon=False, loc="upper right", fontsize=6.8, handlelength=2.5)
    bottom.grid(axis="y", alpha=.18)
    for ax, y in ((top, 0), (bottom, 1)):
        ax.plot([0, 1], [y, y], transform=ax.transAxes, marker=[(-1, -.5), (1, .5)],
                markersize=5, linestyle="none", color=".3", clip_on=False)
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker="o", ms=3, lw=1, label=LABEL[m])
                        for m in methods], loc="lower center", ncol=5, frameon=False,
               columnspacing=1.4, handletextpad=.4, bbox_to_anchor=(.55, -.005))
    fig.subplots_adjust(left=.16, right=.99, top=.88, bottom=.24, wspace=.44)
    save("main_control", fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), sharey=True)
    for j, (ax, key) in enumerate(zip(axes, (ONLINE[0], ONLINE[2]))):
        e = experiments[key]
        q = np.quantile(e["data"]["BBP"]["epsilon"][:, 100:], [.25, .5, .75], axis=0)
        x = np.arange(101, e["plan"]["horizon"] + 1)
        ax.step(x, q[1], where="post", color=COLOR["BBP"], lw=1.5, label="BBP")
        ax.fill_between(x, q[0], q[2], step="post", color=COLOR["BBP"], alpha=.17, lw=0)
        for value, method in ((1e-4, "Small"), (1e-6, "FixedEps1e6")):
            ax.axhline(value, color=COLOR[method], ls=STYLE[method], lw=1.1, label=LABEL[method])
        ax.set(title=f"({chr(97 + j)}) {NAMES[key]}", yscale="log", xlabel="Round",
               xlim=(100, e["plan"]["horizon"]),
               ylim=(min(7e-9, float(q.min()) / 1.2), max(2e-4, float(q.max()) * 1.2)))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel(r"Relative cutoff $\varepsilon_t$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.55, .025))
    fig.subplots_adjust(left=.11, right=.99, top=.90, bottom=.25, wspace=.15)
    save("threshold_dynamics", fig)
    summary["baselines"] = {}
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.linewidth": .65, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.35), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ("v4.exp02", "v4.exp05"),
                             ("(a) Exp. 2: rank 3", "(b) Exp. 5: heavy-tailed noise")):
        e = experiments[key]
        summary["baselines"][key] = {}
        ax.axvline(0, color="#888888", lw=.8, zorder=1)
        for model, offset, marker, color in (("BBP", -.16, "o", "#0072B2"),
                                             ("Full", .16, "s", "#009E73")):
            summary["baselines"][key][model] = {}
            for y, method in enumerate(BASELINES):
                values = (np.asarray(bootstrap(*paired(e, model, method))) - 1) * 100
                summary["baselines"][key][model][method] = values.tolist()
                mid, lo, hi = values
                ax.errorbar(mid, y + offset, xerr=[[mid - lo], [hi - mid]], fmt=marker,
                            markersize=4.5, color=color, elinewidth=1.1, capsize=2.1,
                            capthick=.8, zorder=3)
        ax.set(title=title, xlim=(-60, 30), ylim=(4.65, -.65),
               xticks=[-60, -40, -20, 0, 20], yticks=range(5), yticklabels=BASELINE_LABELS)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=3, width=.65)
        ax.grid(axis="x", color="#DEDEDE", lw=.45, zorder=0)
    fig.subplots_adjust(left=.285, right=.985, top=.865, bottom=.28, wspace=.24)
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=1.1, ms=4.5, label=l)
        for c, m, l in (("#0072B2", "o", "BBP"), ("#009E73", "s", "Full NN-AGP"))],
        loc="lower center", bbox_to_anchor=(.625, .095), ncol=2, frameon=False,
        fontsize=9, columnspacing=2.5)
    fig.supxlabel("Change in mean cumulative regret relative to baseline (%)", x=.625, y=.025, fontsize=9)
    save("baseline_examples", fig)
    plt.rcParams.update(previous_style)
    methods = ["Relative", "Small", "FixedEps1e6", "Eps1e5", "Fixed10", "Fixed25", "BBP"]
    for keys, name, shape, height in ((ONLINE[:6], "appendix_regret_online", (3, 2), 5.75),
                                      (ONLINE[6:], "appendix_regret_v4", (5, 2), 8.5)):
        fig, axes = plt.subplots(*shape, figsize=(7, height))
        used = [m for m in methods if any(m in experiments[k]["data"] for k in keys)]
        for i, (key, ax) in enumerate(zip(keys, np.asarray(axes).flat)):
            e = experiments[key]
            for method in used:
                if method not in e["data"]:
                    continue
                a, b = e["data"][method], e["data"]["Full"]
                if a["seeds"] != b["seeds"]:
                    raise ValueError(f"Unpaired curve: {key}/{method}")
                v = a["regret"] - b["regret"]
                mean = v.mean(0)
                half = t.ppf(.975, len(v) - 1) * v.std(0, ddof=1) / np.sqrt(len(v))
                x = np.arange(1, len(mean) + 1)
                ax.plot(x, mean, color=COLOR[method], ls=STYLE[method],
                        lw=1.4 if method == "BBP" else .9, zorder=3 if method == "BBP" else 2)
                if method == "BBP":
                    ax.fill_between(x, mean - half, mean + half, color=COLOR[method], alpha=.16, lw=0)
            ax.axhline(0, color=".35", lw=.6)
            ax.set(title=f"({chr(97 + i)}) {TITLES[ONLINE.index(key)]}", xlabel="Round",
                   xlim=(1, e["plan"]["horizon"]))
            ax.grid(axis="y", alpha=.16)
            ax.tick_params(labelsize=7)
            ax.title.set_fontsize(8)
            ax.xaxis.label.set_fontsize(7.5)
            if i % shape[1] == 0:
                ax.set_ylabel(r"$\overline{R_t-R_t^{\mathrm{Full}}}$", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=COLOR[m], ls=STYLE[m], lw=1.3, label=LABEL[m])
                            for m in used], loc="lower center", ncol=4, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(.55, .004 if height < 6 else -.014), columnspacing=1.2, handlelength=2.6)
        fig.subplots_adjust(left=.10, right=.99, top=.96, bottom=.15 if height < 6 else .09,
                            hspace=.72, wspace=.30)
        save(name, fig)
    results, samples = phase
    protocol = results["protocol"]
    sizes = np.asarray(protocol["sizes"])
    repetitions = protocol["repetitions"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.95))
    for ax in axes:
        ax.set(xscale="log", xlabel=r"Fresh contexts, $n$", xlim=(sizes.min() * .85, sizes.max() * 1.15))
        ax.grid(axis="y", alpha=.2)
        bounds = [sizes.min(), *protocol["ncrit"], sizes.max()]
        for left, right, label in zip(bounds[:-1], bounds[1:], ("I", "II", "III")):
            ax.axvspan(left, right, color="0.5", alpha=.075 if label == "II" else .035)
            ax.text(np.sqrt(left * right), .98, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=".35")
        for critical in protocol["ncrit"]:
            ax.axvline(critical, ls=":", color="#CC79A7", lw=.9)
    dense = np.geomspace(sizes.min(), sizes.max(), 600)
    gamma = protocol["M"] / dense
    edge = protocol["sigma_eff2"] * (1 + np.sqrt(gamma)) ** 2
    for direction, (delta, color) in enumerate(zip(protocol["spike_ratio"], ("#0072B2", "#009E73"))):
        above = delta > np.sqrt(gamma)
        peak = np.where(above, protocol["sigma_eff2"] * (1 + delta) * (1 + gamma / delta), edge)
        overlap = np.where(above, (1 - gamma / delta ** 2) / (1 + gamma / delta), 0)
        for panel, column, theory in ((0, direction, peak / edge), (1, direction + 2, overlap)):
            values = [samples[f"actual_n{n}"][:, column] / (row["edge"] if panel == 0 else 1)
                      for n, row in zip(sizes, results["observations"])]
            means = np.array([v.mean() for v in values])
            half = np.array([norm.ppf(.975) * v.std(ddof=1) / np.sqrt(repetitions) for v in values])
            controls = [samples[f"classical_Gaussian_n{n}"][:, column].mean() /
                        (row["edge"] if panel == 0 else 1) for n, row in zip(sizes, results["observations"])]
            axes[panel].plot(dense, theory, color=color, ls="--", lw=1.3)
            axes[panel].plot(sizes, controls, color=color, ls=":", lw=1.3)
            axes[panel].fill_between(sizes, means - half, means + half, color=color, alpha=.17, lw=0)
            axes[panel].plot(sizes, means, color=color, marker="o", ms=3.2, lw=1.6)
    axes[0].axhline(1, color=".4", lw=.8)
    axes[0].set(title="(a) Spectral separation", ylabel="Spectral peak / BBP edge")
    axes[1].set(title="(b) Direction recovery", ylabel=r"$|\widehat v_j^\top v_j|^2$", ylim=(-.02, 1.06))
    axes[0].legend(handles=[Line2D([], [], color=c, marker="o", label=f"Direction {j + 1}")
                           for j, c in enumerate(("#0072B2", "#009E73"))],
                   frameon=False, loc="upper left", bbox_to_anchor=(0, .9), fontsize=7)
    fig.legend(handles=[Line2D([], [], color=".3", ls=s, lw=1.3, label=l)
                        for s, l in (("-", "Trained NN-AGP"), ("--", "BBP prediction"),
                                     (":", "Finite Gaussian control"))],
               loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(.53, -.005), handlelength=2.6)
    fig.subplots_adjust(left=.075, right=.99, top=.89, bottom=.25, wspace=.24)
    for ax in axes:
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.tick_params(labelsize=8)
    save("bbp_phases", fig)
    summary["bbp_phase"] = dict(critical_sizes=protocol["ncrit"], repetitions=repetitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, help="Read new runs from this directory")
    parser.add_argument("--output", type=Path, help="Generated tables and figures directory")
    args = parser.parse_args()
    plans = [read(p) for p in sorted(ROOT.glob("*/EXP*/protocol.json"))]
    directory = args.runs.resolve() if args.runs else None
    output = args.output.resolve() if args.output else ROOT / ("paper_rerun" if directory else "paper")
    if output == ROOT or any(output.is_relative_to(ROOT / p["folder"]) for p in plans):
        parser.error("Choose a generated-output directory, not a source/result directory")
    if directory is not None and directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / "reruns"):
        parser.error("New model results inside the package belong under reruns/")
    import run_paper as runner
    hashes = runner.source_hashes()
    for plan in plans:
        for source in plan["sources"]:
            if sha(ROOT / source["file"]) != source["sha256"]:
                raise ValueError(f"Saved source changed: {source['file']}")
    output.mkdir(parents=True, exist_ok=True)
    experiments = load(plans, directory)
    phase_plan = next(p for p in plans if p.get("kind") == "spectral_diagnostic")
    if directory is not None:
        record = result_path(directory, phase_plan, "Trained features", phase_plan["seeds"][0])
        verify_run(read(record), phase_plan, "Trained features", phase_plan["seeds"][0], record)
    phase_dir = (ROOT / phase_plan["folder"] / "results" if directory is None else
                 result_path(directory, phase_plan, "Trained features", phase_plan["seeds"][0]).parent / "classical_transition")
    import numpy as np
    with np.load(phase_dir / "samples.npz", allow_pickle=False) as archive:
        samples = {k: archive[k] for k in archive.files}
    phase = read(phase_dir / "results.json"), samples
    summary = tables(experiments, output)
    with tempfile.TemporaryDirectory(prefix="paper-plots-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["XDG_CACHE_HOME"] = cache
        figures(experiments, phase, output, summary)
    summary["runs"] = sum(len(p["runs"]) for p in plans)
    summary["input"] = "saved" if directory is None else str(directory)
    summary["code_sha256"] = hashes
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), runs=summary["runs"], tables=9, figures=6)))


if __name__ == "__main__":
    main()
