# Paper experiments

`experiments.xlsx` lists settings, averages, and every seed/method.
Each `Environment/EXPxxx/` contains `protocol.json`, `results/`, and `plots/`.
`paper/` contains the nine tables and seven figures used in the article.

Task names map to the existing experiment IDs below. Directory names and raw
environment/method keys are unchanged.

| Task | Main comparison | Other comparisons |
| --- | --- | --- |
| Weak | EXP248 | EXP268 (physical compression) |
| Balanced | EXP001 | EXP018 (physical compression) |
| KuaiRec | EXP095 | EXP104 (physical compression); EXP283 (10 users), EXP284 (100 users), relative cutoff |
| Smooth 12 | EXP121 | |
| Smooth 3 | EXP131 | |
| Smooth 3 (128 inputs) | EXP133 | |
| Exp. 1 | EXP020 | EXP026 (relative cutoff / fixed ranks) |
| Exp. 2 | EXP029 | EXP032 (relative cutoff / fixed ranks) |
| Exp. 3 | EXP035 | EXP040 (relative cutoff / fixed ranks) |
| Exp. 4 | EXP043 | EXP046 (relative cutoff / fixed ranks) |
| Exp. 5 | EXP049 | EXP052 (relative cutoff / fixed ranks) |
| Exp. 6 | EXP055 | EXP058 (relative cutoff / fixed ranks) |
| Exp. 7 | EXP061 | EXP064 (relative cutoff / fixed ranks) |
| Exp. 8 | EXP068 | EXP071 (relative cutoff / fixed ranks) |
| Exp. 9 | EXP075 | EXP079 (relative cutoff / fixed ranks) |
| Exp. 10 | EXP083 | EXP086 (relative cutoff / fixed ranks) |

EXP285 is the BBP transition diagnostic. Method display names are Full, BBP,
Fixed rank 3/5/10/25, Relative cutoff 1e-2/1e-4/1e-5/1e-6, NN-AGP (m=7),
GP-UCB (product), GP-UCB (weighted sum), NeuralUCB, NN-UCB, and
BBP + physical compression. Figures write relative cutoffs with
`epsilon_prune`; plain labels use scientific notation. CLI `--method` accepts
the raw protocol keys (for example, `Fixed10`, `Small`, `Fixed`, `GP mult`,
`GP add`, and `Late`).

Run from the repository root with Python 3.11:

```sh
python -m pip install -r requirements-paper.txt
python plot_paper.py
```

To run an experiment:

```sh
python run_paper.py --list
python run_paper.py --experiment EXP248 --method BBP --seed 6100
```

To retrain everything and rebuild:

```sh
python run_paper.py --all
python plot_paper.py --runs paper_experiments/reruns
```

New runs go to `reruns/`; their tables and figures go to `paper_rerun/`.
Saved results are preserved. Timing depends on hardware.
