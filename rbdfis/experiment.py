"""
Experiment driver for Section 5: ten repetitions of 5-fold cross-validation
(5 x 10 CV) with nested random-search hyperparameter selection.

Each (dataset, model, repetition, fold) task is written to its own JSON file,
so an interrupted run can be resumed and tasks can be parallelized safely.

    python -m rbdfis.experiment --data-dir data --out results
    python -m rbdfis.experiment --data-dir data --out results_R5 --fixed-rules 5   # supplementary analysis
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from .data import DATASETS, load_dataset, make_demo_data
from .models import FuzzyRegressor

# ----------------------------------------------------------------------------
# configuration (Tables 5 and 6)
# ----------------------------------------------------------------------------
MODELS = {
    "TS-0":       {"kind": "ts0"},
    "TS-1":       {"kind": "ts1"},
    "RBD-COG":    {"kind": "cog"},
    "RBD-MOM":    {"kind": "mom"},
    "RBD-WABL-F": {"kind": "wabl", "wabl_params": "fixed"},
    "RBD-WABL-G": {"kind": "wabl", "wabl_params": "global"},
    "RBD-WABL-R": {"kind": "wabl", "wabl_params": "rule"},
}

GRID = {
    "n_rules": [3, 5, 7],
    "lr": [1e-2],
    "ridge": [1e-1],
    "sigma_bar": [0.20],
}

DEFAULTS = dict(n_reps=10, n_folds=5, n_configs=30, inner_val=0.2, max_epochs=200,
                patience=30, batch_size=64, base_seed=2026)


# ----------------------------------------------------------------------------
def task_seed(base, rep, fold):
    return int(base + 1000 * rep + fold)


def sample_configs(grid, n_configs, seed):
    """Same random configurations for every model (Section 5.4)."""
    keys = list(grid)
    full = list(itertools.product(*[grid[k] for k in keys]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(full), size=min(n_configs, len(full)), replace=False)
    return [dict(zip(keys, full[i])) for i in idx]


def project_configs(configs, model_name):
    """TS models do not use sigma_bar: drop it and remove duplicates."""
    if MODELS[model_name]["kind"] in ("ts0", "ts1"):
        seen, out = set(), []
        for c in configs:
            c2 = {k: v for k, v in c.items() if k != "sigma_bar"}
            key = tuple(sorted(c2.items()))
            if key not in seen:
                seen.add(key)
                out.append(c2)
        return out
    return configs


def fit_and_score(model_name, hp, Xtr, ytr, Xte, yte, seed, opts):
    """Scale on the training data only, fit, predict on the original target scale."""
    xs, ys = MinMaxScaler(), MinMaxScaler()
    Xtr_s = xs.fit_transform(Xtr)
    ytr_s = ys.fit_transform(ytr[:, None]).ravel()
    Xte_s = xs.transform(Xte)
    model = FuzzyRegressor(**MODELS[model_name], **hp, max_epochs=opts["max_epochs"],
                           patience=opts["patience"], batch_size=opts["batch_size"], seed=seed,
                           **opts.get("constraints", {}))
    t0 = time.perf_counter()
    model.fit(Xtr_s, ytr_s)
    t_train = time.perf_counter() - t0
    pred = ys.inverse_transform(model.predict(Xte_s)[:, None]).ravel()
    err = yte - pred
    metrics = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err**2) / np.sum((yte - yte.mean()) ** 2)),
        "train_time_s": t_train,
    }
    return model, metrics


def run_task(dataset, model_name, rep, fold, X, y, opts, out_dir):
    path = os.path.join(out_dir, "raw", f"{dataset}__{model_name}__r{rep:02d}_f{fold:02d}.json")
    if os.path.exists(path):
        return path
    seed = task_seed(opts["base_seed"], rep, fold)
    kf = KFold(n_splits=opts["n_folds"], shuffle=True, random_state=opts["base_seed"] + rep)
    tr, te = list(kf.split(X))[fold]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    grid = dict(GRID)
    if opts.get("fixed_rules"):
        grid["n_rules"] = [opts["fixed_rules"]]
    configs = project_configs(sample_configs(grid, opts["n_configs"], seed), model_name)

    # ---- inner model selection (80 % / 20 % of the outer training split)
    if len(configs) > 1:
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=opts["inner_val"], random_state=seed)
        scores = []
        for hp in configs:
            _, m = fit_and_score(model_name, hp, Xtr[itr], ytr[itr], Xtr[iva], ytr[iva], seed, opts)
            scores.append(m["rmse"])
        best_hp = configs[int(np.argmin(scores))]
    else:
        best_hp = configs[0]

    # ---- refit on the whole outer training split, evaluate once on the test fold
    model, met = fit_and_score(model_name, best_hp, Xtr, ytr, Xte, yte, seed, opts)
    rec = {
        "dataset": dataset, "model": model_name, "rep": rep, "fold": fold,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        **met,
        "n_params": model.n_params(X.shape[1]),
        "n_free_params": model.n_free_params(X.shape[1]),
        "epochs": model.n_epochs_,
        **{f"hp_{k}": v for k, v in best_hp.items()},
        **model.wabl_summary(),
        **{f"c_{k}": v for k, v in opts.get("constraints", {}).items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return path


def collect(out_dir):
    raw = os.path.join(out_dir, "raw")
    recs = [json.load(open(os.path.join(raw, f))) for f in sorted(os.listdir(raw)) if f.endswith(".json")]
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all of Table 4")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--reps", type=int, default=DEFAULTS["n_reps"])
    ap.add_argument("--folds", type=int, default=DEFAULTS["n_folds"])
    ap.add_argument("--n-configs", type=int, default=DEFAULTS["n_configs"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["base_seed"])
    ap.add_argument("--fixed-rules", type=int, default=None, help="supplementary analysis with fixed R")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--demo", action="store_true", help="offline smoke test on synthetic data")
    # interpretability constraints (Section 7.7); all off by default
    ap.add_argument("--bound-antecedents", action="store_true",
                    help="keep antecedent centres inside the training range of each input (all models)")
    ap.add_argument("--ordered-consequents", action="store_true",
                    help="RBD models: consequent centres inside the target range and in increasing order")
    ap.add_argument("--b-target", choices=["mean", "partition"], default="mean",
                    help="ridge shrinkage target for the consequent centres")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="minimum gap between ordered consequent centres (fraction of the target range)")
    ap.add_argument("--force-config", action="store_true",
                    help="allow re-using an output folder whose run_config.json differs")
    a = ap.parse_args(argv)

    if a.demo:
        datasets = make_demo_data(a.data_dir)
    else:
        datasets = a.datasets or list(DATASETS)
    opts = dict(n_folds=a.folds, n_configs=a.n_configs, inner_val=DEFAULTS["inner_val"],
                max_epochs=a.max_epochs, patience=a.patience, batch_size=a.batch_size,
                base_seed=a.seed, fixed_rules=a.fixed_rules,
                constraints=dict(bound_antecedents=a.bound_antecedents,
                                 ordered_consequents=a.ordered_consequents,
                                 b_target=a.b_target, min_gap=a.min_gap))
    os.makedirs(a.out, exist_ok=True)
    cfg = {"datasets": datasets, "models": a.models, "reps": a.reps, "grid": GRID, **opts}
    cfg_path = os.path.join(a.out, "run_config.json")
    # guard: never mix results of different settings in one output folder
    # (existing result files are skipped, so they would silently be kept)
    if os.path.exists(cfg_path) and not a.force_config:
        old = json.load(open(cfg_path))
        keys = ("grid", "n_folds", "n_configs", "inner_val", "max_epochs", "patience",
                "batch_size", "base_seed", "fixed_rules", "constraints")
        diff = [k for k in keys if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True)]
        if diff and os.path.isdir(os.path.join(a.out, "raw")) and os.listdir(os.path.join(a.out, "raw")):
            raise SystemExit(f"{a.out} contains results produced with different settings ({', '.join(diff)}). "
                             "Use a new --out folder (or --force-config if you really want to mix them).")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    data = {d: load_dataset(d, a.data_dir) for d in datasets}
    tasks = [(d, m, r, f) for d in datasets for m in a.models for r in range(a.reps) for f in range(a.folds)]
    print(f"{len(tasks)} tasks ({len(datasets)} datasets x {len(a.models)} models x {a.reps} reps x {a.folds} folds)")
    Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_task)(d, m, r, f, *data[d], opts, a.out) for d, m, r, f in tasks)
    df = collect(a.out)
    print(f"collected {len(df)} records -> {os.path.join(a.out, 'results.csv')}")


if __name__ == "__main__":
    main()
