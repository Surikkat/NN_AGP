# Paper experiments

`experiments.xlsx` lists settings, averages, and every seed/method.
Each `Environment/EXPxxx/` contains `protocol.json` and `results/`.
`paper/` contains the nine tables and seven figures used in the article.

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

EXP285 contains the BBP transition and selector experiment.

In each experiment folder:

- `protocol.json`: settings, methods, seeds and result files.
- `results/`: one file per seed and method; BBP checkpoints.

Run from the repository root:

```sh
python run_paper.py --list
python run_paper.py --experiment EXP248 --method BBP --seed 6100
python plot_paper.py
```

Use the method names printed by `--list`; quote names containing spaces.
See the [main README](../README.md) for installation and all-experiment commands.
