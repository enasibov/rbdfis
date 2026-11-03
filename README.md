# RBD-FIS: experiments 

Code for the experimental study of *"An ANFIS-like trainable Mamdani FIS with Gaussian
fuzzy numbers and rule-based defuzzification (RBD) for regression"*.

## Contents

| File | Section | What it does |
|---|---|---|
| `rbdfis/models.py` | 4, 5.2–5.3 | TS-0, TS-1 and RBD-Mamdani (COG, MOM, WABL fixed/global/rule-wise); closed-form rule outputs, analytic gradients, hybrid LSE + Adam learning, early stopping |
| `rbdfis/fcm.py` | 5.3.1 | Fuzzy c-means initialization in the joint input-output space |
| `rbdfis/data.py` | 5.1 | Dataset definitions (10 UCI datasets), UCI download, preprocessing |
| `rbdfis/experiment.py` | 5.4–5.5 | 5 x 10 CV (ten repetitions of 5-fold CV), nested random search (30 configurations, 80/20 inner split), metrics |
| `rbdfis/stats.py` | 5.6 | Friedman + Iman-Davenport, Holm/Finner post-hoc, Nemenyi CD diagrams, Wilcoxon (exact, Holm, rank-biserial), Bayesian signed-rank, corrected resampled t-test, Bayesian correlated t-test |
| `tests/test_core.py` | — | Closed forms vs numerical integration, Propositions 1–3, finite-difference gradient check, stats sanity checks |

## Usage

```bash
pip install -r requirements.txt
python tests/test_core.py                     # all checks should pass

# 1) data: downloads the 10 UCI sets to data/*.csv and prints their shapes.
#    Yacht and QSAR Fish Toxicity are not importable with ucimlrepo and are
#    downloaded directly from the UCI archive. If that fails (proxy/firewall),
#    download the zip from the UCI page, save it as data/yacht.zip or
#    data/qsar_fish.zip, and run the command again. Existing CSVs are kept.
python -m rbdfis.data --fetch --data-dir data
python -m rbdfis.data --check --data-dir data  # compare with Table 4

# 2) main experiment (resumable; one JSON file per task in results/raw)
python -m rbdfis.experiment --data-dir data --out results --n-jobs -1

# 3) supplementary analysis with a fixed number of rules (Section 5.4)
python -m rbdfis.experiment --data-dir data --out results_R5 --fixed-rules 5 --n-jobs -1

# 4) statistics -> analysis/report.md, CSV tables, CD diagrams
python -m rbdfis.stats --results results/results.csv --out analysis

# offline smoke test on synthetic data (reduced budget, a few minutes)
python -m rbdfis.experiment --demo --data-dir data_demo --out res_demo --reps 2 --n-configs 3 --max-epochs 30
python -m rbdfis.stats --results res_demo/results.csv --out an_demo
```

## Notes

* The full run has 10 datasets x 7 models x 10 repetitions x 5 folds = 3,500 tasks, each with
  up to 30 inner fits plus one final fit. Expect tens of CPU-hours; use `--n-jobs` on a multi-core
  machine. Reduce `--n-configs` or `--max-epochs` for pilot runs.
* All randomness is seeded (`--seed`, default 2026). The outer split of repetition r uses
  `KFold(5, shuffle=True, random_state=seed + r)`; the hyperparameter draws, inner split,
  FCM initialization and mini-batch order depend only on (repetition, fold), so every model
  sees identical splits and identical initial antecedents.
* Scaling (min-max) is fitted on the training part of each split, including inside the inner
  model selection; errors are reported on the original target scale.
* The UCI ids, column names and expected sizes in `data.py` were written from memory —
  check the shapes printed by `--fetch`/`--check` before running.
* `results/results.csv` columns: dataset, model, rep, fold, n_train, n_test, rmse, mae, r2,
  train_time_s, n_params, n_free_params, epochs, hp_* (selected hyperparameters), and for WABL
  models cL_mean, cL_std, k_mean, k_std (learned parameters, Section 6.5).

## Interpretability constraints (Section 7.7)

All constraints are off by default, so earlier results are reproduced exactly.

| Option (`FuzzyRegressor` / CLI) | Effect |
|---|---|
| `bound_antecedents=True` / `--bound-antecedents` | Antecedent centres stay inside the training range of each input: m = lo + (hi − lo)·sigmoid(μ). Applies to all models (TS and RBD). `bound_margin` widens the range. |
| `ordered_consequents=True` / `--ordered-consequents` | RBD models: consequent centres stay inside the training range of the target and in non-decreasing order (rules sorted by their initial centre). The least-squares step becomes a small quadratic programme (SLSQP) when the constraints are active. |
| `min_gap=δ` / `--min-gap δ` | With ordered consequents: adjacent centres differ by at least δ·(target range), so the rules stay distinguishable. |
| `b_target="partition"` / `--b-target partition` | The ridge penalty shrinks the consequent centres towards an evenly spaced partition of the target range instead of towards the mean, which prevents their collapse under strong regularization. |

`model.rule_base()` returns the learned centres, widths, consequent centres and spreads
(in the scaled model space) for inspecting or plotting a rule base.

Example (constrained run; use a NEW output folder):

```bash
python -m rbdfis.experiment --data-dir data --out results_constrained \
       --bound-antecedents --ordered-consequents --b-target partition --min-gap 0.1 --n-jobs -1
```

The constraint settings are stored in `run_config.json` and as `c_*` columns in
`results.csv`. The driver now refuses to write into an output folder whose
`run_config.json` has different settings (grid, epochs, seeds, constraints, ...),
because existing result files are skipped and would otherwise be mixed silently;
`--force-config` overrides this check.
