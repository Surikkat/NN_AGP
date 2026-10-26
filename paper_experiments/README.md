# Paper experiments

`experiments.xlsx` lists settings, averages, and every seed/method.
Each `Environment/EXPxxx/` contains `protocol.json`, `results/`, and `plots/`.
`paper/` contains the nine tables and six figures used in the article.

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
