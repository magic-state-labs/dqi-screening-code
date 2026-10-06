"""Script 40 - The log-Cauchy code: a public restricted-error decoder from exponentiation (screen).

The objective selects the error alphabet:
DQI's error register holds vectors on the Fourier support of the per-row objective, so a smooth
objective ``cos(2 pi (Bx)_i / M)`` only ever asks the decoder for ``{0, +-1}`` patterns.  On the
matrix ``B[i][j] = log_g(t_i + alpha_j)`` over a Mersenne prime field ``F_M`` such a pattern
exponentiates to a rational function of degree at most its weight, and rational reconstruction
decodes it (``dqi_explorer/log_cauchy.py``).  The optimization problem: find exponents
``x`` so that ``prod_j (t_i + alpha_j)^{x_j}`` has discrete log near a prescribed centre at as
many points ``t_i`` as possible (multiplicative polynomial intersection; the Chor--Rivest
knapsack with its trapdoor made public).

(A) analytic gates for the cosine objective at ``h in {13, 17, 31}``: restricted D0, cosine D2,
    margin ``sqrt(2 lam(1-lam)) - alpha``, the Gibbs existential line, the restart threshold
    (``R^2 = 4``), buildable length ``m + n <= 2^h``; the finite general-objective value.
(B) built instances: decoder verification with the worst-case all-plus / all-minus patterns,
    the in-measure reach probe (random signs beyond ell), the Schur-square battery against a
    random matrix, and the restricted-decodability (syndrome collision) probe.
(C) what is and is not Reed--Solomon-like about it; the keyed Chor--Rivest variant (class 2 with
    a discrete-log secret; Vaudenay 1998); the lattice view of the classical problem.

Nothing here is an advantage claim.  Run from the repository root:
  python scripts/40_log_cauchy_screen.py [--quick]
"""

from __future__ import annotations

import os
import sys
import time
from math import log2, sqrt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    cosine_baseline,
    cosine_beats_baseline,
    cosine_margin,
    cosine_quality,
    cosine_restart_threshold,
    existential_cosine,
    log_cauchy_reach,
    restricted_qary_entropy,
    restricted_radius_is_realizable,
)
from dqi_explorer.hidden_keys import _square_rank, random_matrix_battery  # noqa: E402
from dqi_explorer.log_cauchy import (  # noqa: E402
    MERSENNE_EXPONENTS,
    build_log_cauchy_instance,
    in_measure_probe,
    restricted_decodability_probe,
    verify_decoder,
)
from dqi_explorer.spectral import general_objective_finite_quality  # noqa: E402

QUICK = "--quick" in sys.argv
ALPHAS = (0.04, 0.05, 0.07, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30)


def section_a() -> None:
    print("=" * 110)
    print("(A) analytic gates, cosine objective: reach alpha/2 (worst case, {0,+-1} errors), baseline alpha")
    print("=" * 110)
    print("  D0 (restricted): alpha >= (H_2(lam) + lam)/log2 M.  D2: lam(1-lam) > alpha^2/2.  D6 void (prime M).")
    print("  Q* = Gibbs bound at alpha ln M nats per row.  threshold = (1-alpha) 4 ln(2^40/0.05)/(2 mu^2).")
    print("  %-3s %-9s %-6s %-8s %-8s %-3s %-3s %-8s %-8s %-8s %-9s %-10s %s" % ("h", "M", "alpha", "lambda", "D0 floor", "D0", "D2", "Q_cos", "Q*", "mu", "threshold", "m+n<=2^h", "verdict"))
    for h in (13, 17, 31):
        M = 2**h - 1
        for alpha in ALPHAS:
            lam = log_cauchy_reach(alpha)
            floor = restricted_qary_entropy(lam, M, 2)
            d0 = restricted_radius_is_realizable(alpha, lam, M)
            d2 = cosine_beats_baseline(alpha, lam)
            q = cosine_quality(lam)
            qstar = existential_cosine(alpha, M) if h <= 17 else float("nan")
            mu = cosine_margin(alpha, lam)
            thr = cosine_restart_threshold(alpha, mu)
            buildable = thr is not None and thr * (1 + alpha) <= 2**h
            verdict = "passes on paper" if (d0 and d2 and buildable) else "; ".join(v for v, bad in (("D0 FAIL", not d0), ("D2 FAIL", not d2), ("threshold > 2^h", not buildable)) if bad)
            print("  %-3d %-9d %-6.2f %-8.4f %-8.4f %-3s %-3s %-8.4f %-8s %+-8.4f %-9.0f %-10s %s" % (h, M, alpha, lam, floor, "ok" if d0 else "no", "ok" if d2 else "no", q, ("%.3f" % qstar) if qstar == qstar else "-", mu, thr or 0, "yes" if buildable else "no", verdict))
    print()
    print("  finite general-objective values (Lemma 9.2 tridiagonal with the cosine moments), h = 13:")
    print("  %-6s %-6s %-5s %-5s %-8s %-8s %-8s %s" % ("m", "alpha", "n", "ell", "Q_fin", "Q_Pr", "mu_fin", "threshold"))
    for m, alpha in ((2000, 0.07), (6000, 0.07), (6000, 0.12), (6000, 0.25)):
        n = int(round(alpha * m))
        ell = (n - 1) // 2
        q = general_objective_finite_quality(m, ell, 0.0, 1 / sqrt(2), 0.0)
        mu = q.value - cosine_baseline(n / m)
        thr = cosine_restart_threshold(n / m, mu)
        print("  %-6d %-6.2f %-5d %-5d %-8.4f %-8.4f %+-8.4f %.0f" % (m, alpha, n, ell, q.value, n / m, mu, thr or 0))


def section_b() -> None:
    print()
    print("=" * 110)
    print("(B) built instances (h = 13, M = 8191)")
    print("=" * 110)
    sizes = [(600, 0.10)] if QUICK else [(600, 0.10), (2000, 0.07)]
    for m, alpha in sizes:
        t0 = time.time()
        inst = build_log_cauchy_instance(m=m, seed=40, h=13, alpha=alpha)
        rng = np.random.default_rng(40)
        print("  m = %d, n = %d, ell = %d, alpha = %.3f (build %.1fs)" % (inst.m, inst.n, inst.ell, inst.alpha, time.time() - t0))
        dec = verify_decoder(inst, 20 if QUICK else 60, 1, beyond_trials=6)
        print("    decoder: %d/%d exact (plus %s, minus %s); beyond radius all-plus at ell+1: %s; %.1fs" % (dec["exact_recoveries"], dec["trials"], dec["by_kind"]["plus"], dec["by_kind"]["minus"], dec["beyond_radius"], dec["seconds"]))
        weights = [inst.ell + k for k in (1, inst.ell // 2, inst.ell, int(1.5 * inst.ell)) if inst.ell + k < inst.n]
        probe = in_measure_probe(inst, weights, 6 if QUICK else 12, 2)
        print("    in-measure probe (random signs): " + ", ".join("w=%d: %.2f" % (w, f) for w, f in probe["recovery_fraction"].items()) + "  (n = %d; the D4' Singleton ceiling is w < n)" % inst.n)
        if m <= 600:
            sq = _square_rank(np.ascontiguousarray(inst.B_pub.T), inst.p, rng)
            ctl = random_matrix_battery(inst.m, inst.n, inst.p, rng, shortenings=(0,), sides=("image",))["image_shorten_0"]
            print("    square of im(B): rank %d / %d (random control %d / %d)" % (sq["square_rank"], sq["random_expectation"], ctl["square_rank"], ctl["random_expectation"]))
        pr = restricted_decodability_probe(inst.B_pub, inst.p, inst.ell, 300, rng)
        print("    restricted-decodability probe: %d collisions in %d random patterns; pigeonhole log2(patterns) = %.0f <= n log2 M = %.0f" % (pr["collisions"], pr["trials"], log2(pr["restricted_patterns"]), inst.n * log2(inst.p)))


def section_c() -> None:
    print()
    print("=" * 110)
    print("(C) reading")
    print("=" * 110)
    print("  Reed-Solomon-like: the decoding algebra (rational reconstruction after exponentiation) and hence the reach")
    print("  alpha/2.  Not OPI: over the DQI alphabet F_M the code has no polynomial structure (entries are logs, square")
    print("  full); the posed problem is a subset-product knapsack with set-valued targets; the objective lives in the log")
    print("  domain and the algebra in the value domain, joined by exponentiation one way and discrete logs the other.")
    print("  Classical hardness: max-cos on a q-ary lattice (n = 420, q = 8191 at the design point), SIS-infinity shaped;")
    print("  script 43 runs lattice reduction on row subsets.  Keyed variant: Chor-Rivest (g, alpha, t secret) is")
    print("  class 2 with a discrete-log secret; Vaudenay (1998) recovers it when h has small factors; not built.")
    print("  Supported Mersenne exponents: %s (tables for 13, 17; 31 and above analytic only)." % (MERSENNE_EXPONENTS,))


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
