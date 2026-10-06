"""Brute-force certificate of the general-objective law  <f>/m = mean + std * lambda_max / m.

Builds the DQI state |P(f)> = sum_x P(f(x)) |x> explicitly on a tiny instance over F_p^n, with

    P(f) = sum_k w_k e_k(g_1(b_1.x), ..., g_m(b_m.x)) / sqrt(p^(n-k) C(m,k)),

g_i the L2-normalised centred objective (Jordan et al., arXiv:2408.08292, Section 8.2), w the
principal eigenvector of the tridiagonal matrix with diagonal k*skew and off-diagonal
sqrt(k (m-k+1)), measures it, and compares the expected objective with the formula for

  (a) an indicator objective   -- Jordan et al. Lemma 9.2 exactly,
  (b) the cosine               -- the smooth-objective law of the paper (supplementary lemma),
  (c) a skewed smooth objective -- the general form.

The formula is exact when the syndrome map is injective on the patterns with entries in the
Fourier support S (plus zero) of weight <= ell together with their one-coordinate S-shifts (for
S = F_p^* this is 2 ell + 1 < d_perp); ``hypothesis_holds`` checks this by enumeration.  A random B
on this tiny instance sometimes violates it, so several seeds are run and the seeds where the
hypothesis holds are the certificate (the norm check alone is weaker and is reported too).  Independent of the library except for the
final comparison with ``general_objective_finite_quality``.

Run from the repository root:  python scripts/44_cosine_law_certificate.py
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dqi_explorer.spectral import general_objective_finite_quality  # noqa: E402


def tridiagonal(m: int, ell: int, skew: float) -> np.ndarray:
    A = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        A[k, k] = k * skew
        A[k - 1, k] = A[k, k - 1] = math.sqrt(k * (m - k + 1))
    return A


def elementary_symmetric(values: np.ndarray, ell: int) -> np.ndarray:
    """e_0..e_ell of the columns of ``values`` (shape m x N) at every point, by the recurrence."""

    m, N = values.shape
    e = np.zeros((ell + 1, N))
    e[0] = 1.0
    for i in range(m):
        for k in range(min(i + 1, ell), 0, -1):
            e[k] += e[k - 1] * values[i]
    return e


def moments(table: np.ndarray) -> tuple[float, float, float]:
    mean = float(table.mean())
    std = float(table.std())
    skew = float(((table - mean) ** 3).mean() / std**3)
    return mean, std, skew


def hypothesis_holds(B: np.ndarray, p: int, ell: int, support: tuple[int, ...]) -> bool:
    """Injectivity of y -> B^T y on weight-<= ell patterns with entries in ``support`` and their shifts."""

    m = B.shape[0]
    syndromes: dict[tuple[int, ...], tuple[int, ...]] = {}
    patterns = []
    for k in range(ell + 1):
        for positions in itertools.combinations(range(m), k):
            for values in itertools.product(support, repeat=k):
                y = np.zeros(m, dtype=np.int64)
                y[list(positions)] = values
                key = tuple((B.T @ y) % p)
                if key in syndromes:
                    return False
                syndromes[key] = tuple(y)
                patterns.append(y)
    for y in patterns:
        for i in range(m):
            for z in support:
                shifted = y.copy()
                shifted[i] = (shifted[i] + z) % p
                key = tuple((B.T @ shifted) % p)
                if key in syndromes and syndromes[key] != tuple(shifted):
                    return False
    return True


def fourier_support(table: np.ndarray) -> tuple[int, ...]:
    p = len(table)
    coeffs = np.fft.fft(table - table.mean())
    return tuple(z for z in range(1, p) if abs(coeffs[z]) > 1e-9)


def certificate(p: int, m: int, n: int, ell: int, F, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    B = rng.integers(0, p, size=(m, n))
    centres = rng.integers(0, p, size=m)
    xs = np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int64)
    args = (xs @ B.T - centres[None, :]) % p  # p^n x m
    table = np.array([F(v, p) for v in range(p)], dtype=float)
    mean, std, skew = moments(table)
    g = (table - mean) / math.sqrt(p) / std  # sum_v g(v)^2 = 1
    e = elementary_symmetric(g[args].T, ell)
    A = tridiagonal(m, ell, skew)
    evals, evecs = np.linalg.eigh(A)
    w = evecs[:, -1]
    coeff = np.array([w[k] / math.sqrt(p ** (n - k) * math.comb(m, k)) for k in range(ell + 1)])
    prob = (coeff @ e) ** 2
    norm = float(prob.sum())
    measured = float((prob * table[args].sum(axis=1)).sum() / norm) / m
    predicted = mean + std * float(evals[-1]) / m
    library = general_objective_finite_quality(m, ell, mean, std, skew).value
    support = fourier_support(table)
    return {"measured": measured, "predicted": predicted, "library": library, "norm_over_w2": norm / float(w @ w),
            "mean": mean, "std": std, "skew": skew, "support": support,
            "hypothesis_holds": hypothesis_holds(B, p, ell, support)}


OBJECTIVES = {
    "indicator r=5": lambda v, p: 1.0 if v in (0, 1, 2, 3, 4) else 0.0,
    "cosine": lambda v, p: math.cos(2 * math.pi * v / p),
    "skewed smooth": lambda v, p: math.cos(2 * math.pi * v / p) + 0.3 * math.cos(4 * math.pi * v / p),
}


SIZES = {
    # the indicator has full Fourier support, so its hypothesis is 2 ell + 1 < d_perp: it needs the
    # larger n (kernel code [8, 2] over F_11, MDS for the seeds run); the smooth objectives have
    # support {+-1} or {+-1, +-2} and certify at n = 5 over F_13
    "indicator r=5": (11, 8, 6, 2),
    "cosine": (13, 8, 5, 2),
    "skewed smooth": (13, 8, 5, 2),
}


def main() -> None:
    print("m=8, ell=2; the formula is exact on the seeds where the injectivity hypothesis holds")
    worst = 0.0
    for name, F in OBJECTIVES.items():
        p, m, n, ell = SIZES[name]
        print("  %s over F_%d^%d:" % (name, p, n))
        for seed in range(1, 7):
            r = certificate(p, m, n, ell, F, seed)
            gap = abs(r["measured"] - r["predicted"])
            if r["hypothesis_holds"]:
                worst = max(worst, gap)
            print("  %-14s seed %d  measured %.12f  formula %.12f  library %.12f  norm/|w|^2 %.10f  %s"
                  % (name, seed, r["measured"], r["predicted"], r["library"], r["norm_over_w2"],
                     "certificate" if r["hypothesis_holds"] else "(hypothesis fails; not a certificate)"))
    print("largest gap on certificate seeds: %.2e" % worst)


if __name__ == "__main__":
    main()
