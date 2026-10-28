# NN–AGP

Code for *Beyond Dimensions: Spectral Compression for Neural–GP Bandits*.

We compress the GP decision space using fixed relative or adaptive BBP
cutoffs while the full encoder continues training. The selected subspace
also supports physical compression of the network output layer.

## Installation

Python 3.11.

```sh
python -m pip install -r requirements-paper.txt
```

## Paper experiments

List experiments and run one:

```sh
python run_paper.py --list
python run_paper.py --experiment EXP248 --method BBP --seed 6100
```

Build the paper's tables and figures from saved results:

```sh
python plot_paper.py
```

Run all experiments and build their tables and figures:

```sh
python run_paper.py --all
python plot_paper.py --runs paper_experiments/reruns
```

Settings, results, figures and the Excel summary are in
[paper_experiments](paper_experiments/README.md).

Model implementations are in `bandits/`, environment generators in
`paper_envs.py`, and the BBP experiment in `paper_bbp.py`.
