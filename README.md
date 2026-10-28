# NN–AGP

Code for *Beyond Dimensions: Spectral Compression for Neural–GP Bandits*.

We compress the GP decision space using fixed relative or adaptive BBP
cutoffs while the full encoder continues training. The selected subspace
also supports physical compression of the network output layer.

![Spectral compression for NN–AGP bandits](experiments/paper/figures/graphical_abstract.svg)

## Installation

Python 3.11.

```sh
python -m pip install -r requirements.txt
```

## Paper experiments

Each experiment's settings are in
`experiments/<environment>/EXPxxx/protocol.json`.

List experiments and run one:

```sh
python -m scripts.run --list
python -m scripts.run --experiment EXP248 --method BBP --seed 6100 --output results/weak
```

Use the method names printed by `--list`; quote names containing spaces.

Build the paper's tables and figures from saved results:

```sh
python -m scripts.plot --output output
```

Run all experiments and build their tables and figures:

```sh
python -m scripts.run --all --output results
python -m scripts.plot --input results --output output
```

Settings, results, figures and the Excel summary are in
[experiments](experiments/README.md).

Model implementations are in `bandits/`. Environment generators are in
`scripts/environments.py`: `WeakBalancedEnv` for Weak and Balanced, `smooth_env` for
Smooth, `kuairec_env` for KuaiRec, and `make_env` with `step` for Exp. 1–10.
The separate BBP experiment is in `scripts/bbp.py`.
