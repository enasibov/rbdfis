"""
Fuzzy regression models used in Section 5.

* TS-0, TS-1 : zero- and first-order Takagi-Sugeno models (ANFIS hybrid learning)
* RBD-COG, RBD-MOM, RBD-WABL : Mamdani FIS with rule-based defuzzification (RBD).
  Every rule's clipped two-sided Gaussian consequent B_i' = min(w_i, B_i) is
  defuzzified individually and the crisp rule outputs are aggregated as
  y_hat = sum_i wbar_i * y_i.

Closed forms used here (E_i = ln(1/w_i), u_i = sqrt(E_i); see Section 4.2):

    y_i = b_i + z_i * phi(E_i)

    MOM  : z_i = sR_i - sL_i,                   phi = u / sqrt(2)
    COG  : z_i = sR_i - sL_i,                   phi = (1 + u^2) / (sqrt(2) u + sqrt(pi/2) erfcx(u))
    WABL : z_i = (1 - cL) sR_i - cL sL_i,       phi = sqrt(2) u + sqrt(pi / (2(k+1))) erfcx(sqrt(k+1) u)

erfcx(x) = exp(x^2) erfc(x) is the scaled complementary error function. The WABL
expression is algebraically identical to
    G(w, k) = sqrt(2) Gamma(3/2, (k+1) ln(1/w)) / (sqrt(k+1) w^(k+1))
(Eq. 6 of the outline), because Gamma(3/2, s) = sqrt(s) e^-s + (sqrt(pi)/2) erfc(sqrt(s)).
This form needs no special overflow handling.

All gradients are analytic (no autodiff dependency); tests/test_core.py checks
them against finite differences.

Interpretability constraints (optional, off by default so that earlier results
are reproduced exactly):

* bound_antecedents=True   : every antecedent centre is kept inside the range of
  its input in the training data, m_ij = lo_j + (hi_j - lo_j) * sigmoid(mu_ij).
* ordered_consequents=True : (RBD models only) the consequent centres are kept
  inside the range of the training target and in non-decreasing order,
  y_lo <= b_1 <= b_2 <= ... <= b_R <= y_hi. The rules are sorted by their initial
  consequent centre, and the least-squares step becomes a small quadratic
  programme with these linear constraints.
* b_target='partition'     : the ridge penalty shrinks the consequent centres
  towards an evenly spaced partition of the target range instead of towards the
  mean target ('mean', the default), which counteracts the collapse of the
  centres under strong regularization.
* min_gap=delta            : with ordered_consequents, adjacent consequent centres
  differ by at least delta * (y_hi - y_lo), which keeps the rules distinguishable.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.special import erfcx, expit

from .fcm import fcm

SQRT2 = np.sqrt(2.0)
SQRT_PI = np.sqrt(np.pi)
SQRT_PI_2 = np.sqrt(np.pi / 2.0)
EPS_WIDTH = 1e-6


# ----------------------------------------------------------------------------
# elementary helpers
# ----------------------------------------------------------------------------
def softplus(x):
    return np.logaddexp(0.0, x)


def softplus_inv(y):
    y = np.asarray(y, dtype=float)
    return y + np.log(-np.expm1(-y))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


# ----------------------------------------------------------------------------
# per-rule shape functions phi(E) and their derivatives
# ----------------------------------------------------------------------------
def phi_mom(E):
    u = np.sqrt(E)
    with np.errstate(divide="ignore"):
        dE = 1.0 / (2.0 * SQRT2 * u)
    return u / SQRT2, dE


def phi_cog(E):
    u = np.sqrt(E)
    ex = erfcx(u)
    D = SQRT2 * u + SQRT_PI_2 * ex
    phi = (1.0 + E) / D
    dE = (D - (1.0 + E) * SQRT_PI_2 * ex) / D**2
    return phi, dE


def phi_wabl(E, k):
    kap = k + 1.0
    sk = np.sqrt(kap)
    u = np.sqrt(E)
    v = sk * u
    ex = erfcx(v)
    phi = SQRT2 * u + np.sqrt(np.pi / (2.0 * kap)) * ex
    dE = np.sqrt(np.pi * kap / 2.0) * ex
    dk = SQRT_PI_2 / (2.0 * kap) * (-ex / sk + 2.0 * u * v * ex - 2.0 * u / SQRT_PI)
    return phi, dE, dk


def rule_output(method, b, sL, sR, w, k=1.0, cL=0.5):
    """Defuzzified value of the clipped two-sided Gaussian min(w, G(b, sL, sR))
    in the notation of the paper (used by the tests and for reporting)."""
    E = -np.log(w)
    if method == "mom":
        return b + (sR - sL) * phi_mom(E)[0]
    if method == "cog":
        return b + (sR - sL) * phi_cog(E)[0]
    if method == "wabl":
        return b + ((1.0 - cL) * sR - cL * sL) * phi_wabl(E, k)[0]
    raise ValueError(method)


# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for key, g in grads.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g * g
            mh = self.m[key] / (1 - self.b1**self.t)
            vh = self.v[key] / (1 - self.b2**self.t)
            params[key] = params[key] - self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# main model
# ----------------------------------------------------------------------------
class FuzzyRegressor:
    """ANFIS-like fuzzy regressor.

    kind        : 'ts0' | 'ts1' | 'cog' | 'mom' | 'wabl'
    wabl_params : 'fixed' | 'global' | 'rule'   (only for kind='wabl')
    Inputs and target are expected to be scaled to [0, 1] (done by the
    experiment driver).
    """

    def __init__(self, kind="wabl", n_rules=5, lr=5e-3, ridge=1e-3, sigma_bar=0.1,
                 wabl_params="global", cL0=0.5, k0=1.0, max_epochs=500, patience=30,
                 batch_size=64, val_frac=0.2, s_min=0.05, E_min=1e-6, fcm_m=2.0,
                 spread_margin=1e-3, seed=0, bound_antecedents=False,
                 ordered_consequents=False, b_target="mean", bound_margin=0.0,
                 min_gap=0.0):
        if kind not in ("ts0", "ts1", "cog", "mom", "wabl"):
            raise ValueError(kind)
        if b_target not in ("mean", "partition"):
            raise ValueError(b_target)
        self.bound_antecedents = bool(bound_antecedents)
        self.ordered_consequents = bool(ordered_consequents)
        self.b_target = b_target
        self.bound_margin = float(bound_margin)
        self.min_gap = float(min_gap)
        if self.min_gap * (n_rules - 1) >= 1.0:
            raise ValueError("min_gap * (n_rules - 1) must be < 1")
        self.kind = kind
        self.n_rules = int(n_rules)
        self.lr, self.ridge, self.sigma_bar = lr, ridge, sigma_bar
        self.wabl_params = wabl_params
        self.cL0, self.k0 = cL0, k0
        self.max_epochs, self.patience = max_epochs, patience
        self.batch_size, self.val_frac = batch_size, val_frac
        self.s_min, self.E_min, self.fcm_m = s_min, E_min, fcm_m
        self.spread_margin = spread_margin
        self.seed = seed

    # ---------------------------------------------------------------- props
    @property
    def is_rbd(self):
        return self.kind in ("cog", "mom", "wabl")

    @property
    def trains_wabl(self):
        return self.kind == "wabl" and self.wabl_params in ("global", "rule")

    def _cL(self, p=None):
        p = self.p if p is None else p
        return expit(p["gamma_c"])

    def _k(self, p=None):
        p = self.p if p is None else p
        return softplus(p["theta_k"])

    def _z(self, p=None):
        """Effective consequent parameter z_i (Proposition 5)."""
        if self.kind == "wabl":
            c = np.broadcast_to(self._cL(p), (self.n_rules,))
            return (1.0 - c) * self.sR - c * self.sL
        return self.sR - self.sL

    def _centres(self, p=None):
        """Antecedent centres (bounded reparametrization if bound_antecedents)."""
        p = self.p if p is None else p
        if self.bound_antecedents:
            return self.x_lo + self.x_rng * expit(p["m"])
        return p["m"]

    def _b_partition(self):
        """Evenly spaced reference centres over the target range."""
        R = self.n_rules
        return self.y_lo + (np.arange(R) + 0.5) / R * (self.y_hi - self.y_lo)

    # ---------------------------------------------------------------- init
    def _init(self, X, y, rng):
        R, n = self.n_rules, X.shape[1]
        V, U = fcm(np.column_stack([X, y]), R, m=self.fcm_m, rng=rng)
        Um = U**self.fcm_m                                   # (R, N)
        centres = V[:, :n]
        var = np.einsum("rt,rtj->rj", Um, (X[None] - centres[:, None]) ** 2) / Um.sum(1, keepdims=True)
        widths = np.maximum(np.sqrt(var), self.s_min)
        b0 = V[:, n].copy()
        # domain bounds (training data), optionally widened by bound_margin
        x_min, x_max = X.min(axis=0), X.max(axis=0)
        xr = np.where(x_max > x_min, x_max - x_min, 1.0)
        self.x_lo = x_min - self.bound_margin * xr
        self.x_rng = xr * (1.0 + 2.0 * self.bound_margin)
        self.y_lo, self.y_hi = float(y.min()), float(y.max())
        if self.ordered_consequents and self.is_rbd:
            order = np.argsort(b0)                         # rule i <-> i-th smallest centre
            centres, widths, b0 = centres[order], widths[order], b0[order]
        if self.bound_antecedents:
            u = np.clip((centres - self.x_lo) / self.x_rng, 1e-3, 1.0 - 1e-3)
            m_param = logit(u)
        else:
            m_param = centres.copy()
        self.p = {"m": m_param, "rho": softplus_inv(widths - EPS_WIDTH)}
        self.b = b0
        if self.kind == "ts1":
            self.B = np.zeros((R, n + 1))
            self.B[:, 0] = self.b
        if self.is_rbd:
            self.sL = np.full(R, self.sigma_bar)
            self.sR = np.full(R, self.sigma_bar)
        if self.kind == "wabl":
            nw = R if self.wabl_params == "rule" else 1
            self.p["gamma_c"] = np.full(nw, logit(self.cL0))
            self.p["theta_k"] = np.full(nw, softplus_inv(self.k0))

    # ---------------------------------------------------------------- forward
    def _forward(self, X, p=None):
        p = self.p if p is None else p
        s = softplus(p["rho"]) + EPS_WIDTH                     # (R, n)
        d = X[:, None, :] - self._centres(p)[None]             # (N, R, n)
        q = d / s[None]
        E = 0.5 * np.einsum("trj,trj->tr", q, q)               # (N, R) = ln(1/w)
        a = -E - (-E).max(axis=1, keepdims=True)
        w = np.exp(a)
        wbar = w / w.sum(axis=1, keepdims=True)
        out = {"d": d, "s": s, "E": E, "wbar": wbar}
        if self.kind == "ts0":
            yr = np.broadcast_to(self.b, E.shape)
        elif self.kind == "ts1":
            Xa = np.column_stack([np.ones(len(X)), X])
            yr = Xa @ self.B.T
            out["Xa"] = Xa
        else:
            # clamp w <= 1 - E_min (upper clamp on firing strength, Section 5.3.2)
            Ec = np.maximum(E, self.E_min)
            active = (E > self.E_min).astype(float)
            if self.kind == "mom":
                phi, dphi = phi_mom(Ec)
            elif self.kind == "cog":
                phi, dphi = phi_cog(Ec)
            else:
                k = self._k(p)[None, :]
                phi, dphi, dk = phi_wabl(Ec, k)
                out["dk"] = dk
            out["phi"], out["dphi"] = phi, dphi * active
            z = self._z(p)
            out["z"] = z
            yr = self.b[None] + z[None] * phi
        out["yr"] = yr
        out["yhat"] = np.sum(wbar * yr, axis=1)
        return out

    # ---------------------------------------------------------------- gradients
    def _grads(self, X, y, p=None):
        """Gradients of L = 0.5 * mean((y_hat - y)^2) w.r.t. the nonlinear parameters."""
        p = self.p if p is None else p
        f = self._forward(X, p)
        g = (f["yhat"] - y) / len(y)                           # dL/dy_hat
        wbar = f["wbar"]
        gE = g[:, None] * wbar * (f["yhat"][:, None] - f["yr"])
        if self.is_rbd:
            gE = gE + g[:, None] * wbar * f["z"][None] * f["dphi"]
        s, d = f["s"], f["d"]
        g_centre = np.einsum("tr,trj->rj", gE, -d) / s**2
        if self.bound_antecedents:
            sg = expit(p["m"])
            g_centre = g_centre * self.x_rng * sg * (1.0 - sg)
        grads = {
            "m": g_centre,
            "rho": np.einsum("tr,trj->rj", gE, -(d * d)) / s**3 * expit(p["rho"]),
        }
        if self.trains_wabl:
            gk_rule = np.sum(g[:, None] * wbar * f["z"][None] * f["dk"], axis=0)
            gc_rule = np.sum(g[:, None] * wbar * f["phi"], axis=0) * (-(self.sL + self.sR))
            if self.wabl_params == "global":
                gk_rule, gc_rule = gk_rule.sum(keepdims=True), gc_rule.sum(keepdims=True)
            c = self._cL(p)
            grads["theta_k"] = gk_rule * expit(p["theta_k"])
            grads["gamma_c"] = gc_rule * c * (1.0 - c)
        loss = 0.5 * np.mean((f["yhat"] - y) ** 2)
        return grads, loss

    # ---------------------------------------------------------------- LSE step
    def _lse(self, X, y):
        f = self._forward(X)
        W, n, R = f["wbar"], len(y), self.n_rules
        ybar = y.mean()
        yc = y - ybar                                         # ridge shrinks towards the mean
        lam = self.ridge * n
        if self.kind == "ts0":
            A = W
        elif self.kind == "ts1":
            A = (W[:, :, None] * f["Xa"][:, None, :]).reshape(n, -1)
        else:
            A = np.hstack([W, W * f["phi"]])
        if not self.is_rbd:
            theta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yc)
            if self.kind == "ts0":
                self.b = theta + ybar
            else:
                self.B = theta.reshape(R, -1)
                self.B[:, 0] += ybar
            return
        # bounded ridge LSE for (b_i, z_i); bounds keep both spreads positive
        sb, mg = self.sigma_bar, 1.0 - self.spread_margin
        if self.kind == "wabl":
            c = np.broadcast_to(self._cL(), (R,))
            lo_z, hi_z = -2.0 * c * sb * mg, 2.0 * (1.0 - c) * sb * mg
        else:
            lo_z, hi_z = np.full(R, -2.0 * sb * mg), np.full(R, 2.0 * sb * mg)
        # ridge target for the (centred) consequent centres
        t = (self._b_partition() - ybar) if self.b_target == "partition" else np.zeros(R)
        # substitute b = t + b' so that the penalty acts on b' (shrinkage towards t)
        yc_t = yc - W @ t
        lo = np.concatenate([np.full(R, -np.inf), lo_z])
        hi = np.concatenate([np.full(R, np.inf), hi_z])
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(2 * R)])
        y_aug = np.concatenate([yc_t, np.zeros(2 * R)])
        theta = lsq_linear(A_aug, y_aug, bounds=(lo, hi), method="bvls").x
        theta[:R] += t
        if self.ordered_consequents:
            theta = self._ordered_qp(A, yc, lam, t, theta, lo_z, hi_z, ybar)
        self.b = theta[:R] + ybar
        z = theta[R:]
        if self.kind == "wabl":                              # sR + sL = 2 sb, (1-c) sR - c sL = z
            self.sR = z + 2.0 * c * sb
        else:                                                 # sR - sL = z
            self.sR = sb + z / 2.0
        self.sL = 2.0 * sb - self.sR

    def _ordered_qp(self, A, yc, lam, t, theta0, lo_z, hi_z, ybar):
        """min ||A th - yc||^2 + lam ||b - t||^2 + lam ||z||^2 subject to
        y_lo <= b_1 <= ... <= b_R <= y_hi (centred by ybar) and lo_z <= z <= hi_z."""
        R = self.n_rules
        b_lo, b_hi = self.y_lo - ybar, self.y_hi - ybar
        gap = self.min_gap * (self.y_hi - self.y_lo)
        b0 = theta0[:R]
        if np.all(np.diff(b0) >= gap) and b0[0] >= b_lo and b0[-1] <= b_hi:
            return theta0                                     # constraints inactive
        H = A.T @ A + lam * np.eye(2 * R)
        g = A.T @ yc + lam * np.concatenate([t, np.zeros(R)])
        # linear inequalities G th >= h
        G = np.zeros((R + 1, 2 * R))
        h = np.zeros(R + 1)
        for i in range(R - 1):
            G[i, i], G[i, i + 1] = -1.0, 1.0                  # b_{i+1} - b_i >= gap
            h[i] = gap
        G[R - 1, 0], h[R - 1] = 1.0, b_lo                     # b_1 >= y_lo
        G[R, R - 1], h[R] = -1.0, -b_hi                       # -b_R >= -y_hi
        start = theta0.copy()
        start[:R] = b_lo + gap * np.arange(R) + np.clip(np.sort(b0) - b_lo - gap * np.arange(R),
                                                        0.0, b_hi - b_lo - gap * (R - 1))  # feasible start
        start[:R] = np.maximum.accumulate(start[:R] - gap * np.arange(R)) + gap * np.arange(R)
        bounds = [(None, None)] * R + list(zip(lo_z, hi_z))
        res = minimize(lambda th: 0.5 * th @ H @ th - g @ th, start,
                       jac=lambda th: H @ th - g, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": lambda th: G @ th - h, "jac": lambda th: G}],
                       options={"ftol": 1e-12, "maxiter": 500})
        th = res.x
        # remove tiny numerical violations
        k = gap * np.arange(R)
        th[:R] = np.clip(np.maximum.accumulate(th[:R] - k), b_lo, b_hi - gap * (R - 1)) + k
        th[R:] = np.clip(th[R:], lo_z, hi_z)
        return th

    def rule_base(self):
        """Learned fuzzy sets in the (scaled) model space, rules in model order."""
        out = {"centres": self._centres().copy(),
               "widths": softplus(self.p["rho"]) + EPS_WIDTH,
               "b": np.asarray(self.b).copy()}
        if self.is_rbd:
            out.update(sL=self.sL.copy(), sR=self.sR.copy())
        return out

    # ---------------------------------------------------------------- state
    def _state(self):
        keys = ("p", "b", "B", "sL", "sR")
        return {k: copy.deepcopy(getattr(self, k)) for k in keys if hasattr(self, k)}

    def _restore(self, st):
        for k, v in st.items():
            setattr(self, k, v)

    # ---------------------------------------------------------------- public API
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        nv = max(1, int(round(self.val_frac * len(y))))
        va, tr = idx[:nv], idx[nv:]
        Xt, yt, Xv, yv = X[tr], y[tr], X[va], y[va]

        self._init(Xt, yt, rng)
        self._lse(Xt, yt)
        best, best_state, wait = self._rmse(Xv, yv), self._state(), 0
        opt = Adam(self.lr)
        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            perm = rng.permutation(len(yt))
            for st in range(0, len(yt), self.batch_size):
                bi = perm[st:st + self.batch_size]
                grads, _ = self._grads(Xt[bi], yt[bi])
                opt.step(self.p, grads)
            self._lse(Xt, yt)                                 # consequents optimal for current antecedents
            r = self._rmse(Xv, yv)
            if r < best - 1e-12:
                best, best_state, wait = r, self._state(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self._restore(best_state)
        self.n_epochs_, self.best_val_rmse_ = epoch, best
        return self

    def predict(self, X):
        return self._forward(np.asarray(X, float))["yhat"]

    def _rmse(self, X, y):
        return float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))

    # ---------------------------------------------------------------- reporting
    def n_params(self, n_inputs):
        """Parameter count as defined in Section 5.4 (Table 3)."""
        R, n = self.n_rules, n_inputs
        base = 2 * n * R
        if self.kind == "ts0":
            return base + R
        if self.kind == "ts1":
            return base + R * (n + 1)
        extra = 0
        if self.kind == "wabl" and self.wabl_params == "global":
            extra = 2
        elif self.kind == "wabl" and self.wabl_params == "rule":
            extra = 2 * R
        return base + 3 * R + extra

    def n_free_params(self, n_inputs):
        """Free parameters after the total-spread constraint (one spread per rule removed)."""
        p = self.n_params(n_inputs)
        return p - self.n_rules if self.is_rbd else p

    def wabl_summary(self):
        if self.kind != "wabl":
            return {}
        c, k = np.atleast_1d(self._cL()), np.atleast_1d(self._k())
        return {"cL_mean": float(c.mean()), "cL_std": float(c.std()),
                "k_mean": float(k.mean()), "k_std": float(k.std())}
