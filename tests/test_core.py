"""Run with `python -m pytest tests` or `python tests/test_core.py`."""
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc, gamma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbdfis.models import FuzzyRegressor, phi_wabl, rule_output  # noqa: E402
from rbdfis import stats as S  # noqa: E402

CASES = [(0.3, 0.7, 1.6, 0.4, 2.0, 0.3), (-1.0, 1.2, 0.5, 0.9, 0.0, 0.7),
         (2.0, 0.4, 0.4, 0.05, 5.0, 0.5), (0.0, 0.2, 2.5, 0.999, 1.0, 0.1)]


def mu_clipped(y, b, sL, sR, w):
    s = sL if y <= b else sR
    return min(w, np.exp(-(y - b) ** 2 / (2 * s * s)))


def test_cog_mom_wabl_closed_forms():
    for b, sL, sR, w, k, cL in CASES:
        lim = 12 * max(sL, sR)
        num = quad(lambda y: y * mu_clipped(y, b, sL, sR, w), b - lim, b + lim, points=[b], limit=400)[0]
        den = quad(lambda y: mu_clipped(y, b, sL, sR, w), b - lim, b + lim, points=[b], limit=400)[0]
        assert abs(rule_output("cog", b, sL, sR, w) - num / den) < 1e-6
        a = np.sqrt(2 * np.log(1 / w))
        mom_ref = ((b - sL * a) + (b + sR * a)) / 2
        assert abs(rule_output("mom", b, sL, sR, w) - mom_ref) < 1e-10
        p = lambda al: (k + 1) * al**k
        L = lambda al: b - sL * np.sqrt(-2 * np.log(al))
        R = lambda al: b + sR * np.sqrt(-2 * np.log(al))
        wabl_ref = quad(lambda al: (cL * L(al) + (1 - cL) * R(al)) * p(al), 0, w)[0] / quad(p, 0, w)[0]
        assert abs(rule_output("wabl", b, sL, sR, w, k, cL) - wabl_ref) < 1e-6


def test_erfcx_form_equals_incomplete_gamma_form():
    for w in [0.9, 0.5, 0.1, 0.01]:
        for k in [0.0, 1.0, 4.0]:
            s = (k + 1) * np.log(1 / w)
            G = np.sqrt(2) * gammaincc(1.5, s) * gamma(1.5) / (np.sqrt(k + 1) * w ** (k + 1))
            assert abs(phi_wabl(np.log(1 / w), k)[0] - G) < 1e-9


def test_propositions():
    b, s, w = 0.4, 0.8, 0.3
    # Proposition 2: symmetric consequent -> COG = MOM = b
    assert abs(rule_output("cog", b, s, s, w) - b) < 1e-12
    assert abs(rule_output("mom", b, s, s, w) - b) < 1e-12
    # Proposition 3: WABL -> MOM for cL = 1/2, k -> infinity
    assert abs(rule_output("wabl", b, 0.5, 1.5, w, 1e7, 0.5) - rule_output("mom", b, 0.5, 1.5, w)) < 1e-3
    # Proposition 1: scaled (product-implication) consequent -> WABL independent of w
    def wabl_scaled(w, sL=0.5, sR=1.5, k=2.0, cL=0.3):
        p = lambda al: (k + 1) * al**k
        h = lambda al: np.sqrt(2 * np.log(w / al))
        f = lambda al: (cL * (b - sL * h(al)) + (1 - cL) * (b + sR * h(al))) * p(al)
        return quad(f, 0, w)[0] / quad(p, 0, w)[0]
    assert abs(wabl_scaled(0.2) - wabl_scaled(0.8)) < 1e-6


def test_gradients_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.random((40, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    y = (y - y.min()) / (y.max() - y.min())
    for kind, wp in [("ts0", None), ("ts1", None), ("cog", None), ("mom", None),
                     ("wabl", "global"), ("wabl", "rule")]:
        m = FuzzyRegressor(kind=kind, n_rules=4, wabl_params=wp or "global", sigma_bar=0.2, seed=0)
        m._init(X, y, rng)
        m._lse(X, y)
        if m.is_rbd:     # asymmetric spreads so that every term is active
            m.sR = m.sR + 0.05 * rng.random(4)
            m.sL = 0.4 - m.sR
        grads, _ = m._grads(X, y)
        for key, g in grads.items():
            num = np.zeros_like(g)
            for idx in np.ndindex(g.shape):
                old = m.p[key][idx]
                m.p[key][idx] = old + 1e-6
                lp = m._grads(X, y)[1]
                m.p[key][idx] = old - 1e-6
                lm = m._grads(X, y)[1]
                m.p[key][idx] = old
                num[idx] = (lp - lm) / 2e-6
            err = np.max(np.abs(num - g)) / (np.max(np.abs(num)) + 1e-12)
            assert err < 1e-4, (kind, wp, key, err)


def test_training_reduces_error():
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.sin(4 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 2]
    y = (y - y.min()) / (y.max() - y.min())
    base = np.sqrt(np.mean((y - y.mean()) ** 2))
    for kind in ["ts0", "ts1", "cog", "mom", "wabl"]:
        m = FuzzyRegressor(kind=kind, n_rules=5, max_epochs=60, seed=0).fit(X, y)
        assert np.sqrt(np.mean((m.predict(X) - y) ** 2)) < 0.6 * base, kind


def test_stats_helpers():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(S.holm(p), [0.04, 0.09, 0.09, 0.5])
    q, cd = S.nemenyi_cd(4, 10)
    assert abs(q - 2.569) < 2e-3 and abs(cd - 1.483) < 2e-3
    q, cd = S.nemenyi_cd(5, 10)
    assert abs(q - 2.728) < 2e-3 and abs(cd - 1.929) < 2e-3
    probs = S.bayesian_signed_rank(np.full(10, -0.05), n_samples=5000)
    assert probs[0] > 0.95


def test_gradients_with_bounded_antecedents():
    rng = np.random.default_rng(3)
    X = rng.random((40, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    y = (y - y.min()) / (y.max() - y.min())
    for kind, wp in [("ts1", "global"), ("cog", "global"), ("wabl", "rule")]:
        m = FuzzyRegressor(kind=kind, n_rules=4, wabl_params=wp, sigma_bar=0.2, seed=0,
                           bound_antecedents=True, ordered_consequents=True)
        m._init(X, y, rng)
        m._lse(X, y)
        grads, _ = m._grads(X, y)
        g = grads["m"]
        num = np.zeros_like(g)
        for idx in np.ndindex(g.shape):
            old = m.p["m"][idx]
            m.p["m"][idx] = old + 1e-6
            lp = m._grads(X, y)[1]
            m.p["m"][idx] = old - 1e-6
            lm = m._grads(X, y)[1]
            m.p["m"][idx] = old
            num[idx] = (lp - lm) / 2e-6
        assert np.max(np.abs(num - g)) / (np.max(np.abs(num)) + 1e-12) < 1e-4, kind


def test_constraints_hold_after_training():
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.sin(4 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 2]
    y = (y - y.min()) / (y.max() - y.min())
    for kind in ["cog", "mom", "wabl"]:
        for lam, tgt in [(1e-1, "mean"), (1e-1, "partition"), (1e-4, "mean")]:
            m = FuzzyRegressor(kind=kind, n_rules=5, ridge=lam, max_epochs=40, seed=0,
                               bound_antecedents=True, ordered_consequents=True,
                               b_target=tgt, min_gap=0.05).fit(X, y)
            c = m.rule_base()["centres"]
            assert np.all(c >= m.x_lo - 1e-12) and np.all(c <= m.x_lo + m.x_rng + 1e-12)
            b = np.asarray(m.b)
            gap = 0.05 * (m.y_hi - m.y_lo)
            assert np.all(np.diff(b) >= gap - 1e-9), (kind, lam, tgt, b)
            assert b[0] >= m.y_lo - 1e-9 and b[-1] <= m.y_hi + 1e-9
            assert np.all(m.sL > 0) and np.all(m.sR > 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
