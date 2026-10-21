# NN–AGP

Spectral compression uses the joint features `encoder(contexts) @ B`.
The full network keeps training; GP predictions use the selected directions.

Install: `python -m pip install numpy scipy torch`.

Initialize once; call `select_action` and `update` each round with your environment’s data:

```python
from bandits.nn_agp_adaptive import NN_AGP_UCB_Adaptive

agent = NN_AGP_UCB_Adaptive(theta_dim=6, x_dim=1, prune_rule="bbp", seed=0)
x, mean, variance = agent.select_action(actions, context)
agent.update(context, x, reward)
```

- `bbp`: estimates the cutoff from the spectral background; keeps Full if no reliable peak is found.
- `relative` (default): fixed relative cutoff, e.g. `eps_prune=1e-4` (default `0.01`).
- `none`: full GP.

Defaults: 100 outputs, 64 hidden units, ten Adam steps per observation at `lr=1e-3`.
Fitting uses at most 256 sampled observations; GP conditioning uses the full history.
Computations use float64. Compression starts after 100 observations and repeats every ten.
`beta=None` uses the v5 UCB schedule; a number sets a constant beta.
`init_noise` is the observation-noise standard deviation; its variance is learned.

`eps_prune` applies to joint covariance eigenvalues, rather than encoder singular values.
Calibration uses recorded contexts; fresh contexts can be passed as `calibration_contexts` to `update`.
