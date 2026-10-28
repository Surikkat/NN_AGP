# Paper experiments

`experiments.xlsx` lists settings, averages, and every seed/method.
Each environment's experiment folders contain `protocol.json` and `results/`.
Generators are in [scripts/environments.py](../scripts/environments.py).

| Environment | Description | Experiments |
| --- | --- | --- |
| Weak | Context baseline and two unequal action preferences | EXP248 (main); EXP268 (physical compression) |
| Balanced | Context baseline and two equal-strength action preferences | EXP001 (main); EXP018 (physical compression) |
| KuaiRec | Video recommendation using clipped watch ratios and learned user/video features | EXP095 (main); EXP104 (physical compression); EXP283 (relative cutoff, 10 users); EXP284 (relative cutoff, 100 users) |
| Smooth 12 | Smooth rewards built from 12 latent factors | EXP121 (main) |
| Smooth 3 | Smooth rewards built from three latent factors | EXP131 (main) |
| Smooth 3 (128 inputs) | Three smooth factors with 123 irrelevant context coordinates | EXP133 (main) |
| Exp. 1 | Five signal factors plus 95 very weak factors | EXP020 (main); EXP026 (relative cutoff / fixed ranks) |
| Exp. 2 | Three signal factors with randomly masked context coordinates | EXP029 (main); EXP032 (relative cutoff / fixed ranks) |
| Exp. 3 | Ten signal factors plus 40 randomly weighted weak factors | EXP035 (main); EXP040 (relative cutoff / fixed ranks) |
| Exp. 4 | Eight signal factors with saturating nonlinear rewards | EXP043 (main); EXP046 (relative cutoff / fixed ranks) |
| Exp. 5 | Sparse factor loadings and reward noise with heavy tails | EXP049 (main); EXP052 (relative cutoff / fixed ranks) |
| Exp. 6 | Active signal dimension increasing from three to eight | EXP055 (main); EXP058 (relative cutoff / fixed ranks) |
| Exp. 7 | Five stable factors plus five with oscillating strengths | EXP061 (main); EXP064 (relative cutoff / fixed ranks) |
| Exp. 8 | Active signal dimension expanding from five to fifteen, then returning | EXP068 (main); EXP071 (relative cutoff / fixed ranks) |
| Exp. 9 | Five signal factors with continuously rotating action directions | EXP075 (main); EXP079 (relative cutoff / fixed ranks) |
| Exp. 10 | Five signal factors with an abrupt rotation at round 100 | EXP083 (main); EXP086 (relative cutoff / fixed ranks) |

EXP285 contains the BBP transition and selector experiment. Its settings,
training and sampling are in [scripts/bbp.py](../scripts/bbp.py); `results/`
contains the checkpoint, calibration and sampled spectra.

In each online experiment folder:

- `protocol.json`: environment and model settings, methods, seeds and result paths.
- `results/`: one JSON file per seed and method, with per-round actions, rewards,
  regret and selected dimensions.

See the [main README](../README.md) for installation and run commands.
