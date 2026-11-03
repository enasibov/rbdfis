"""Fuzzy c-means clustering (Bezdek et al., 1984) for rule-base initialization (Section 5.3.1)."""
import numpy as np


def fcm(Z, c, m=2.0, tol=1e-5, max_iter=300, rng=None):
    """Return prototypes V (c, d) and memberships U (c, N)."""
    rng = np.random.default_rng() if rng is None else rng
    N = Z.shape[0]
    U = rng.random((c, N))
    U /= U.sum(axis=0, keepdims=True)
    expo = 2.0 / (m - 1.0)
    for _ in range(max_iter):
        Um = U**m
        V = (Um @ Z) / Um.sum(axis=1, keepdims=True)
        D = np.sqrt(((Z[None, :, :] - V[:, None, :]) ** 2).sum(axis=2)) + 1e-12   # (c, N)
        inv = D ** (-expo)
        U_new = inv / inv.sum(axis=0, keepdims=True)
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new
    Um = U**m
    V = (Um @ Z) / Um.sum(axis=1, keepdims=True)
    return V, U
