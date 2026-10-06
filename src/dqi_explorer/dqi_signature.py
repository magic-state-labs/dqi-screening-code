"""Values ``P(f)`` of the DQI polynomial (``dqi_polynomial_values``) and the distribution of
the agreement count under the DQI state (``agreement_distribution``), whose mean reproduces
the finite Lemma 9.2 value.  Used by script 38, section (C), and by ``screen.dqi_value``.
"""

from __future__ import annotations

from math import comb, lgamma, log, log2, sqrt

import numpy as np

from .spectral import canonical_finite_quality


def _log_binom(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def dqi_polynomial_values(m: int, p: int, r: int, ell: int) -> np.ndarray:
    """``P(f)`` for ``f = 0..m``, normalized so that ``E[P(f)^2] = 1`` under the binomial model.

    The DQI state is ``sum_k w_k |P_k>`` with ``|P_k> = C(m,k)^(-1/2) sum_x e_k(g(x)) |x>``
    where ``g_i(x) = (1[(Bx)_i in F_i] - rho) / sigma`` and ``e_k`` is the ``k``-th elementary
    symmetric polynomial.  With ``f`` satisfied rows, ``e_k`` depends only on ``f``:
    ``e_k(f) = (r (p-r))^(-k/2) sum_j C(f,j) C(m-f,k-j) (p-r)^j (-r)^(k-j)``.  The inner sum
    is computed exactly in Python integers and rescaled in the log domain.
    """

    quality = canonical_finite_quality(m, p, r, ell)
    weights = np.asarray(quality.coefficients, dtype=np.float64)  # w_0 .. w_ell (unit norm)
    if ell == 0:
        return np.ones(m + 1) * weights[0]
    log2_scale = np.array([0.5 * k * log2(r * (p - r)) + 0.5 * _log_binom(m, k) / log(2) for k in range(ell + 1)])
    values = np.zeros(m + 1)
    pr = p - r
    for f in range(m + 1):
        A = [comb(f, j) * pr**j for j in range(ell + 1)]
        Bc = [comb(m - f, j) * (-r) ** j for j in range(ell + 1)]
        total = 0.0
        for k in range(ell + 1):
            E = 0
            for j in range(k + 1):
                E += A[j] * Bc[k - j]
            if E == 0:
                continue
            sign = 1.0 if E > 0 else -1.0
            mag = abs(E)
            bits = mag.bit_length()
            mantissa = (mag >> max(0, bits - 53)) / float(1 << min(bits, 53))
            log2_term = bits + log2(mantissa) - log2_scale[k]
            total += weights[k] * sign * 2.0**log2_term
        values[f] = total
    return values


def agreement_distribution(m: int, p: int, r: int, ell: int) -> dict:
    """``Pr[f]`` under ``Pr[x] ∝ P(f(x))^2`` with binomial level counts; mean checked against Lemma 9.2."""

    P = dqi_polynomial_values(m, p, r, ell)
    rho = r / p
    log_w = np.array([_log_binom(m, f) + f * log(rho) + (m - f) * log(1 - rho) for f in range(m + 1)])
    with np.errstate(divide="ignore"):
        log_pf = log_w + 2 * np.log(np.abs(P))
    log_pf[~np.isfinite(log_pf)] = -np.inf
    probs = np.exp(log_pf - log_pf.max())
    probs /= probs.sum()
    f = np.arange(m + 1)
    mean = float((probs * f).sum())
    lemma = canonical_finite_quality(m, p, r, ell).value * m
    return {
        "m": m,
        "p": p,
        "r": r,
        "ell": ell,
        "probabilities": probs,
        "mean": mean,
        "mean_fraction": mean / m,
        "lemma92_mean": lemma,
        "abs_diff": abs(mean - lemma),
        "std": float(sqrt((probs * (f - mean) ** 2).sum())),
        "polynomial": P,
    }


__all__ = [
    "agreement_distribution",
    "dqi_polynomial_values",
]
