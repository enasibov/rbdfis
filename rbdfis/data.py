"""
Benchmark datasets of Table 4.

Every dataset is stored as  <data_dir>/<name>.csv  (numeric inputs first, target
in the LAST column, with a header row). `python -m rbdfis.data --fetch` creates
these files from the UCI repository using the `ucimlrepo` package; datasets
that ucimlrepo cannot import (Yacht, QSAR Fish Toxicity) are downloaded
directly from the UCI archive (see DIRECT_SOURCES). Existing CSV files are
kept, so --fetch can be re-run safely. A KEEL .dat file placed in <data_dir> is also
accepted and converted automatically (optional).

IMPORTANT: the UCI ids, column names and expected sizes below were written from
memory. The fetch step prints the resulting shapes; compare them with
EXPECTED_SHAPE (and with Table 4) before running the experiments.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DATASETS = {
    # name           label                              UCI id  expected (N, n)
    "airfoil":     {"label": "Airfoil Self-Noise",          "uci_id": 291, "shape": (1503, 5)},
    "auto_mpg":    {"label": "Auto MPG",                    "uci_id": 9,   "shape": (392, 6)},
    "ccpp":        {"label": "Combined Cycle Power Plant",  "uci_id": 294, "shape": (9568, 4)},
    "concrete":    {"label": "Concrete Compressive Strength", "uci_id": 165, "shape": (1030, 8)},
    "energy":      {"label": "Energy Efficiency (heating)", "uci_id": 242, "shape": (768, 8)},
    "real_estate": {"label": "Real Estate Valuation",       "uci_id": 477, "shape": (414, 6)},
    "wine_red":    {"label": "Wine Quality (red)",          "uci_id": 186, "shape": (1599, 11)},
    "yacht":       {"label": "Yacht Hydrodynamics",         "uci_id": 243, "shape": (308, 6)},
    "abalone":     {"label": "Abalone",                     "uci_id": 1,   "shape": (4177, 7)},
    "qsar_fish":   {"label": "QSAR Fish Toxicity",          "uci_id": 504, "shape": (908, 6)},
}

# categorical / identifier columns that are removed (Section 5.1)
DROP_COLUMNS = {
    "auto_mpg": ["origin", "car_name"],
    "abalone": ["Sex"],
    "real_estate": ["No"],
}


# ----------------------------------------------------------------------------
def _postprocess(name, X: pd.DataFrame, y: pd.DataFrame, original: pd.DataFrame | None):
    if name == "wine_red":
        # the UCI 'wine quality' entry contains red and white wines
        if original is not None and "color" in original.columns:
            mask = (original["color"].astype(str).str.lower() == "red").values
            X, y = X[mask], y[mask]
        else:
            raise RuntimeError("Could not identify red wines; download winequality-red.csv manually.")
    if name == "energy":
        y = y.iloc[:, [0]]                      # Y1 = heating load
    X = X.drop(columns=[c for c in DROP_COLUMNS.get(name, []) if c in X.columns])
    df = pd.concat([X.reset_index(drop=True), y.iloc[:, [0]].reset_index(drop=True)], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


# Direct-download fallback for datasets that `ucimlrepo` cannot import
# (e.g. Yacht Hydrodynamics). Files are plain numeric tables without header,
# target in the last column. URLs are tried in order.
DIRECT_SOURCES = {
    "yacht": {
        "n_cols": 7,
        "urls": [
            "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        ],
    },
    "qsar_fish": {
        "n_cols": 7,
        "urls": [
            "https://archive.ics.uci.edu/static/public/504/qsar+fish+toxicity.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv",
        ],
    },
}
RAW_EXTENSIONS = (".data", ".csv", ".txt", ".dat")


def _parse_numeric_table(text, n_cols):
    """Keep every line that splits into exactly n_cols numbers (separators: ; , whitespace)."""
    import re

    rows = []
    for line in text.splitlines():
        parts = [p for p in re.split(r"[;,\s]+", line.strip()) if p]
        if len(parts) != n_cols:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue                       # header or malformed line
    if not rows:
        raise ValueError(f"no rows with {n_cols} numeric columns found")
    return pd.DataFrame(rows)


def _table_from_bytes(raw, n_cols):
    """raw: bytes of a .zip archive or of a plain text table."""
    import io
    import zipfile

    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(RAW_EXTENSIONS)]
            for m in members:
                try:
                    return _parse_numeric_table(zf.read(m).decode("utf-8", errors="ignore"), n_cols)
                except ValueError:
                    continue
        raise ValueError("no usable table inside the zip archive")
    return _parse_numeric_table(raw.decode("utf-8", errors="ignore"), n_cols)


def _fetch_direct(name, data_dir):
    """Try a manually downloaded file first (data/<name>.zip|.data|.csv|.txt), then the URLs."""
    import urllib.request

    src = DIRECT_SOURCES[name]
    for ext in (".zip",) + RAW_EXTENSIONS:
        local = os.path.join(data_dir, name + ext)
        if ext != ".csv" and os.path.exists(local):
            with open(local, "rb") as fh:
                return _table_from_bytes(fh.read(), src["n_cols"])
    errors = []
    for url in src["urls"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return _table_from_bytes(r.read(), src["n_cols"])
        except Exception as e:             # try the next URL
            errors.append(f"{url}: {e}")
    raise RuntimeError("direct download failed:\n  " + "\n  ".join(errors))


def fetch_uci(data_dir="data", force=False):
    os.makedirs(data_dir, exist_ok=True)
    failed = []
    for name, meta in DATASETS.items():
        path = os.path.join(data_dir, f"{name}.csv")
        if os.path.exists(path) and not force:
            df = pd.read_csv(path)
            print(f"{name:12s} exists {(df.shape[0], df.shape[1] - 1)}  expected {meta['shape']}  (skipped)")
            continue
        try:
            if name in DIRECT_SOURCES:
                df = _fetch_direct(name, data_dir)
                how = "direct"
            else:
                from ucimlrepo import fetch_ucirepo   # pip install ucimlrepo
                ds = fetch_ucirepo(id=meta["uci_id"])
                original = getattr(ds.data, "original", None)
                df = _postprocess(name, ds.data.features, ds.data.targets, original)
                how = "ucimlrepo"
        except Exception as e:
            failed.append(name)
            print(f"{name:12s} FAILED: {e}")
            continue
        df.to_csv(path, index=False)
        exp = meta["shape"]
        got = (df.shape[0], df.shape[1] - 1)
        flag = "OK" if got == exp else "CHECK!"
        print(f"{name:12s} saved {got}  expected {exp}  {flag}  [{how}]")
    if failed:
        print("\nNot downloaded: " + ", ".join(failed))
        print("Download the file(s) manually from https://archive.ics.uci.edu, save as "
              "data/<name>.zip (or .data/.txt), and run --fetch again.")


def read_keel(path):
    """Parse a KEEL .dat file (last attribute = output)."""
    rows, in_data = [], False
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if line.startswith("@"):
                continue
            if in_data:
                rows.append([float(v) for v in line.replace(" ", "").split(",")])
    return pd.DataFrame(rows)


def load_dataset(name, data_dir="data"):
    """Return X (N, n), y (N,) as float arrays."""
    csv = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(csv):
        dat = os.path.join(data_dir, f"{name}.dat")
        if os.path.exists(dat):
            read_keel(dat).to_csv(csv, index=False)
        else:
            raise FileNotFoundError(f"{csv} (or {dat}) not found; run `python -m rbdfis.data --fetch`.")
    df = pd.read_csv(csv).apply(pd.to_numeric, errors="coerce").dropna()
    X = df.iloc[:, :-1].to_numpy(float)
    y = df.iloc[:, -1].to_numpy(float)
    return X, y


def make_demo_data(data_dir="data_demo", seed=0):
    """Small synthetic regression problems for an offline smoke test."""
    from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

    os.makedirs(data_dir, exist_ok=True)
    gens = {"friedman1": make_friedman1, "friedman2": make_friedman2, "friedman3": make_friedman3}
    rng = np.random.default_rng(seed)
    names = []
    for i in range(4):
        for gname, gen in gens.items():
            if len(names) >= 10:
                break
            kw = {"n_features": 5} if gname == "friedman1" else {}
            X, y = gen(n_samples=int(rng.integers(200, 400)), noise=0.1 * (i + 1), random_state=seed + i, **kw)
            name = f"{gname}_{i}"
            pd.DataFrame(np.column_stack([X, y])).to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
            names.append(name)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the UCI datasets")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--check", action="store_true", help="print shapes of the local files")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = ap.parse_args()
    if a.fetch:
        fetch_uci(a.data_dir, force=a.force)
    if a.check:
        for nm, meta in DATASETS.items():
            try:
                X, y = load_dataset(nm, a.data_dir)
                print(f"{nm:12s} {X.shape}  expected {meta['shape']}")
            except FileNotFoundError as e:
                print(f"{nm:12s} missing ({e})")
