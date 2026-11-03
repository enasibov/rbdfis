"""
Statistical analysis of Section 5.6.

    python -m rbdfis.stats --results results/results.csv --out analysis

Produces (CSV + a Markdown report):
  * mean +- sd of RMSE, MAE, R2 per dataset and model, parameter counts, times
  * Family A / Family B: average ranks, Friedman + Iman-Davenport,
    post-hoc z-tests (Holm, Finner), Nemenyi critical difference + CD diagram
  * pairwise Wilcoxon signed-rank tests on relative RMSE differences
    (exact p-values, Holm, matched-pairs rank-biserial correlation)
  * Bayesian signed-rank test (ROPE, Dirichlet-process prior)
  * per dataset: corrected resampled t-test (Nadeau-Bengio) and
    Bayesian correlated t-test on the 50 paired folds
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats as st

FAMILY_A = ["TS-0", "TS-1", "RBD-WABL-G", "RBD-WABL-R"]
FAMILY_B = ["RBD-COG", "RBD-MOM", "RBD-WABL-F", "RBD-WABL-G", "RBD-WABL-R"]
CONTROL = "RBD-WABL-G"
KEY_PAIRS = [(CONTROL, m) for m in ("TS-0", "TS-1", "RBD-COG", "RBD-MOM")]
ALPHA = 0.05
ROPE = 0.01                    # +-1 % relative RMSE
N_MC = 50_000


# ----------------------------------------------------------------------------
# multiple-comparison adjustments
# ----------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float)
    m, order = len(p), np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def finner(p):
    p = np.asarray(p, float)
    m, order = len(p), np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order, start=1):
        run = max(run, 1.0 - (1.0 - p[j]) ** (m / i))
        adj[j] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------
# across-dataset tests
# ----------------------------------------------------------------------------
def friedman_iman_davenport(M: pd.DataFrame):
    """M: datasets x models matrix of mean RMSE (lower is better)."""
    N, k = M.shape
    ranks = M.rank(axis=1, method="average")
    R = ranks.mean(axis=0)
    chi2 = 12.0 * N / (k * (k + 1)) * (np.sum(R**2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = st.chi2.sf(chi2, k - 1)
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
    p_ff = st.f.sf(FF, k - 1, (k - 1) * (N - 1))
    return {"N": N, "k": k, "avg_ranks": R, "chi2_F": chi2, "p_chi2": p_chi2, "F_F": FF, "p_FF": p_ff}


def nemenyi_cd(k, N, alpha=ALPHA):
    q = st.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return q, q * np.sqrt(k * (k + 1) / (6.0 * N))


def posthoc_rank_tests(R: pd.Series, N, pairs):
    k = len(R)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    rows = []
    for a, b in pairs:
        z = (R[a] - R[b]) / se
        rows.append({"A": a, "B": b, "rank_A": R[a], "rank_B": R[b], "z": z, "p": 2 * st.norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["p_finner"] = finner(df["p"])
    return df


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    r = st.rankdata(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return (tp - tm) / (tp + tm) if tp + tm > 0 else 0.0


def bayesian_signed_rank(d, rope=ROPE, prior=1.0, n_samples=N_MC, seed=0):
    """Benavoli et al. (2017): returns P(left), P(rope), P(right).
    With d = (RMSE_A - RMSE_B)/RMSE_B, 'left' means A is practically better."""
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], np.asarray(d, float)])
    alpha = np.concatenate([[prior], np.ones(len(d))])
    W = rng.dirichlet(alpha, n_samples)
    S = z[:, None] + z[None, :]
    regions = [(S < -2 * rope), (np.abs(S) <= 2 * rope), (S > 2 * rope)]
    theta = np.stack([np.einsum("si,ij,sj->s", W, Rg.astype(float), W) for Rg in regions], axis=1)
    winner = np.argmax(theta, axis=1)
    return np.bincount(winner, minlength=3) / n_samples


def wilcoxon_pairs(M: pd.DataFrame, pairs, seed=0):
    rows = []
    for a, b in pairs:
        d = ((M[a] - M[b]) / M[b]).to_numpy()
        has_ties = len(np.unique(np.abs(d[d != 0]))) < np.count_nonzero(d)
        method = "exact" if (np.all(d != 0) and not has_ties) else "approx"
        res = st.wilcoxon(d, alternative="two-sided", method=method)
        pl, pr, pg = bayesian_signed_rank(d, seed=seed)
        rows.append({"A": a, "B": b, "median_rel_diff": float(np.median(d)),
                     "W": float(res.statistic), "p": float(res.pvalue), "method": method,
                     "r_rb": rank_biserial(d),
                     "P(A better)": pl, "P(rope)": pr, "P(B better)": pg})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["bayes_decision"] = np.select(
        [df["P(A better)"] > 0.95, df["P(rope)"] > 0.95, df["P(B better)"] > 0.95],
        ["A better", "equivalent", "B better"], default="undecided")
    return df


# ----------------------------------------------------------------------------
# per-dataset tests
# ----------------------------------------------------------------------------
def corrected_resampled_t(d, test_train_ratio):
    d = np.asarray(d, float)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / n + test_train_ratio) * var)
    return t, 2 * st.t.sf(abs(t), n - 1)


def bayesian_correlated_t(d, rho, rope=ROPE):
    d = np.asarray(d, float)
    n = len(d)
    scale = np.sqrt((1.0 / n + rho / (1.0 - rho)) * d.var(ddof=1))
    if scale == 0:
        return (0.0, 1.0, 0.0)
    dist = st.t(df=n - 1, loc=d.mean(), scale=scale)
    pl = dist.cdf(-rope)
    pg = dist.sf(rope)
    return pl, 1.0 - pl - pg, pg


def per_dataset_tests(df: pd.DataFrame, pairs):
    rows = []
    for a, b in pairs:
        sub = []
        for ds, g in df.groupby("dataset"):
            ga = g[g.model == a].set_index(["rep", "fold"]).sort_index()
            gb = g[g.model == b].set_index(["rep", "fold"]).sort_index()
            idx = ga.index.intersection(gb.index)
            if len(idx) < 3:
                continue
            ra, rb = ga.loc[idx, "rmse"].to_numpy(), gb.loc[idx, "rmse"].to_numpy()
            ratio = float((ga.loc[idx, "n_test"] / ga.loc[idx, "n_train"]).mean())
            rho = float((ga.loc[idx, "n_test"] / (ga.loc[idx, "n_test"] + ga.loc[idx, "n_train"])).mean())
            t, p = corrected_resampled_t(ra - rb, ratio)
            pl, pr, pg = bayesian_correlated_t((ra - rb) / rb, rho)
            sub.append({"dataset": ds, "A": a, "B": b, "n_pairs": len(idx), "mean_diff": float((ra - rb).mean()),
                        "t": t, "p": p, "P(A better)": pl, "P(rope)": pr, "P(B better)": pg})
        sub = pd.DataFrame(sub)
        if len(sub):
            sub["p_holm"] = holm(sub["p"])
            sub["outcome"] = np.where(sub["p_holm"] >= ALPHA, "tie",
                                      np.where(sub["mean_diff"] < 0, "win", "loss"))
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------
# critical-difference diagram
# ----------------------------------------------------------------------------
def cd_diagram(avg_ranks: pd.Series, cd, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = avg_ranks.sort_values()
    k = len(R)
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.32 * k))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    y0 = 0.85
    ax.hlines(y0, 1, k, color="k", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, y0, y0 + 0.03, color="k", lw=1)
        ax.text(r, y0 + 0.05, str(r), ha="center", va="bottom", fontsize=9)
    ax.hlines(y0 + 0.17, 1, 1 + cd, color="k", lw=2)
    ax.text(1 + cd / 2, y0 + 0.19, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    half = (k + 1) // 2
    step = 0.7 / max(half, 1)
    for i, (name, r) in enumerate(R.items()):
        left = i < half
        yy = y0 - 0.12 - step * (i if left else (k - 1 - i))
        xe = 0.6 if left else k + 0.4
        ax.plot([r, r, xe], [y0, yy, yy], color="k", lw=0.8)
        ax.text(xe + (-0.05 if left else 0.05), yy, f"{name} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)
    # cliques: maximal groups whose rank range < CD
    vals = R.to_numpy()
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yy = y0 - 0.05 - 0.04 * c
        ax.hlines(yy, vals[i] - 0.03, vals[j] + 0.03, color="tab:red", lw=3)
    ax.text((k + 1) / 2, 1.17, title, ha="center", va="top", fontsize=10)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def summary_tables(df):
    agg = df.groupby(["dataset", "model"]).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        n_params=("n_params", "median"), train_time_s=("train_time_s", "mean"),
        n_outer_folds=("rmse", "size")).reset_index()
    return agg


def analyse_family(name, models, M, out, report, pairs_mode):
    M = M[models].dropna()
    fr = friedman_iman_davenport(M)
    q, cd = nemenyi_cd(fr["k"], fr["N"])
    report.append(f"\n## Family {name}: {', '.join(models)}\n")
    report.append(f"N = {fr['N']} datasets, k = {fr['k']} models\n")
    report.append("Average ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in fr["avg_ranks"].sort_values().items()) + "\n")
    report.append(f"Friedman chi2_F = {fr['chi2_F']:.3f} (p = {fr['p_chi2']:.4g}); "
                  f"Iman-Davenport F_F = {fr['F_F']:.3f} (p = {fr['p_FF']:.4g})\n")
    report.append(f"Nemenyi q_0.05 = {q:.3f}, CD = {cd:.3f}\n")
    if pairs_mode == "control":
        pairs = [(CONTROL, m) for m in models if m != CONTROL]
    else:
        pairs = list(itertools.combinations(models, 2))
    ph = posthoc_rank_tests(fr["avg_ranks"], fr["N"], pairs)
    wx = wilcoxon_pairs(M, pairs)
    ph.to_csv(os.path.join(out, f"family{name}_posthoc_ranks.csv"), index=False)
    wx.to_csv(os.path.join(out, f"family{name}_wilcoxon_bayes.csv"), index=False)
    fr["avg_ranks"].rename("avg_rank").to_csv(os.path.join(out, f"family{name}_avg_ranks.csv"))
    cd_diagram(fr["avg_ranks"], cd, os.path.join(out, f"family{name}_cd.png"), f"Family {name}")
    report.append("\nPost-hoc rank tests:\n\n" + ph.round(4).to_markdown(index=False) + "\n")
    report.append("\nWilcoxon signed-rank and Bayesian signed-rank tests (d = (RMSE_A - RMSE_B)/RMSE_B):\n\n"
                  + wx.round(4).to_markdown(index=False) + "\n")
    return fr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.results)

    agg = summary_tables(df)
    agg.to_csv(os.path.join(a.out, "summary_by_dataset_model.csv"), index=False)
    M = agg.pivot(index="dataset", columns="model", values="rmse_mean")
    M.to_csv(os.path.join(a.out, "mean_rmse_matrix.csv"))

    report = ["# Statistical analysis (Section 5.6)\n",
              f"Records: {len(df)}; outer folds per dataset/model: "
              f"{int(agg['n_outer_folds'].min())}-{int(agg['n_outer_folds'].max())}\n",
              "\n## Mean test RMSE (datasets x models)\n\n" + M.round(4).to_markdown() + "\n"]
    present = set(M.columns)
    if set(FAMILY_A) <= present:
        analyse_family("A", FAMILY_A, M, a.out, report, "control")
    if set(FAMILY_B) <= present:
        analyse_family("B", FAMILY_B, M, a.out, report, "all")
    pairs = [p for p in KEY_PAIRS if set(p) <= present]
    pdt = per_dataset_tests(df, pairs)
    pdt.to_csv(os.path.join(a.out, "per_dataset_tests.csv"), index=False)
    if len(pdt):
        wtl = pdt.groupby(["A", "B"])["outcome"].value_counts().unstack(fill_value=0)
        report.append("\n## Per-dataset corrected resampled t-tests: win/tie/loss of A vs B (Holm over datasets)\n\n"
                      + wtl.to_markdown() + "\n")
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
