"""Script 42 - Other one-way maps from the exponents: is there another mechanism?

DQI's error register lives on the objective's Fourier support, so under a smooth objective a
decoder for ``{0, +-1}`` patterns suffices.  The exponent knapsack (script 40) works because the
one-way isomorphism ``(F_M, +) -> (F_(2^h)^*, x)`` turns such a pattern into a rational function
of low degree that ``n`` evaluations identify.  A fourth mechanism would be another one-way
isomorphism from the additive alphabet into a group where small-integer combinations are
identifiable.  This script records each candidate switch with the reason it closes or the
generator that opens it (``log_cauchy.switch_closures``), and screens the two that open:

  (A) the switch table;
  (B) the safe-prime family: ``B_ij = log_(g^2)((t_i + alpha_j)^2)`` over ``F_M``, ``M = (q-1)/2``,
      ``q`` a safe prime; images are squares of a rational function, reach ``alpha/4``; gates,
      decoder verification with the worst cases, in-measure probe, square battery;
  (C) the mixed key: the exponent knapsack under a secret dense change of basis ``T``; the key
      holder's decoder, the exp-affine key-recovery problem with its counting bound, and why the
      key-less controls of script 41 apply verbatim (same code, same objective);
  (D) two uses: the keyless verifiable challenge (public form) and hash-and-sign on the mixed
      key, with the union-bound forgery numbers for the cosine objective (``R = 2``).

Nothing here is an advantage claim.  Run from the repository root:
  python scripts/42_domain_switch_sweep.py [--quick]
"""

from __future__ import annotations

import os
import sys
import time
from math import exp, log, log2, sqrt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    cosine_baseline,
    cosine_beats_baseline,
    cosine_margin,
    cosine_quality,
    cosine_restart_threshold,
    existential_cosine,
    restricted_qary_entropy,
    restricted_radius_is_realizable,
)
from dqi_explorer.hidden_keys import _square_rank, random_matrix_battery  # noqa: E402
from dqi_explorer.log_cauchy import (  # noqa: E402
    build_mixed_log_cauchy_instance,
    build_safe_prime_instance,
    exp_affine_pair_test,
    in_measure_probe,
    next_safe_prime,
    switch_closures,
    verify_decoder,
)
from dqi_explorer.spectral import general_objective_finite_quality  # noqa: E402

QUICK = "--quick" in sys.argv


def section_a() -> None:
    print("=" * 110)
    print("(A) domain switches: one-way isomorphisms from the additive alphabet, and whether small patterns are identifiable")
    print("=" * 110)
    for row in switch_closures():
        print("  - %s" % row["switch"])
        print("      identification: %s" % row["identification"])
        print("      reach: %s | status: %s" % (row["reach"], row["status"]))
    print("  Reading: the exp/log switch is the one isomorphism with a degree structure on the image side; its two forms")
    print("  (characteristic 2, safe primes) are the third mechanism, and a secret basis on top of it is a key on that mechanism.")


def section_b() -> None:
    print()
    print("=" * 110)
    print("(B) the safe-prime family: reach alpha/4, cosine objective, alphabet M = (q-1)/2")
    print("=" * 110)
    print("  %-7s %-7s %-6s %-8s %-8s %-3s %-3s %-8s %-8s %-8s %s" % ("q", "M", "alpha", "lambda", "D0 floor", "D0", "D2", "Q_cos", "Q*", "mu", "threshold"))
    for q in (next_safe_prime(4001), next_safe_prime(12001), next_safe_prime(60000)):
        M = (q - 1) // 2
        for alpha in (0.07, 0.12, 0.25):
            lam = alpha / 4
            print("  %-7d %-7d %-6.2f %-8.4f %-8.4f %-3s %-3s %-8.4f %-8.3f %+-8.4f %.0f" % (q, M, alpha, lam, restricted_qary_entropy(lam, M, 2), "ok" if restricted_radius_is_realizable(alpha, lam, M) else "no", "ok" if cosine_beats_baseline(alpha, lam) else "no", cosine_quality(lam), existential_cosine(alpha, M) if M <= 2**17 else float("nan"), cosine_margin(alpha, lam), cosine_restart_threshold(alpha, cosine_margin(alpha, lam)) or 0))
    print("  finite values (Lemma 9.2 tridiagonal with the cosine moments), m = 6000: " + "; ".join("alpha=%.2f: Q_fin %.4f vs %.2f" % (a, general_objective_finite_quality(6000, (int(round(a * 6000)) - 1) // 4, 0.0, 1 / sqrt(2), 0.0).value, a) for a in (0.07, 0.12, 0.25)))
    sizes = [(600, 0.10)] if QUICK else [(600, 0.10), (2000, 0.12)]
    for m, alpha in sizes:
        t0 = time.time()
        inst = build_safe_prime_instance(m=m, seed=42, alpha=alpha)
        rng = np.random.default_rng(42)
        print("  m = %d, q = %d, M = %d, n = %d, ell = %d (build %.1fs)" % (inst.m, inst.q, inst.p, inst.n, inst.ell, time.time() - t0))
        dec = verify_decoder(inst, 16 if QUICK else 40, 1, beyond_trials=6)
        print("    decoder: %d/%d exact (plus %s, minus %s); all-plus at ell+1: %s; %.1fs" % (dec["exact_recoveries"], dec["trials"], dec["by_kind"]["plus"], dec["by_kind"]["minus"], dec["beyond_radius"], dec["seconds"]))
        weights = [w for w in (inst.ell + 1, 2 * inst.ell, 3 * inst.ell, int(0.9 * inst.n)) if w < inst.n]
        probe = in_measure_probe(inst, weights, 6 if QUICK else 10, 2)
        print("    in-measure probe (random signs): " + ", ".join("w=%d: %.2f" % (w, f) for w, f in probe["recovery_fraction"].items()) + "  (n = %d)" % inst.n)
        if m <= 600:
            sq = _square_rank(np.ascontiguousarray(inst.B_pub.T), inst.p, rng)
            ctl = random_matrix_battery(inst.m, inst.n, inst.p, rng, shortenings=(0,), sides=("image",))["image_shorten_0"]
            print("    square of im(B): rank %d / %d (random control %d / %d)" % (sq["square_rank"], sq["random_expectation"], ctl["square_rank"], ctl["random_expectation"]))


def section_c() -> None:
    print()
    print("=" * 110)
    print("(C) the mixed key: B_pub = B T, T a secret dense change of basis over F_M")
    print("=" * 110)
    m, alpha = (600, 0.10) if QUICK else (2000, 0.07)
    inst = build_mixed_log_cauchy_instance(m=m, seed=42, h=13, alpha=alpha)
    dec = verify_decoder(inst, 12 if QUICK else 30, 1, beyond_trials=4)
    print("  m = %d, n = %d, ell = %d: key holder's decoder %d/%d exact; all-plus at ell+1 %s" % (inst.m, inst.n, inst.ell, dec["exact_recoveries"], dec["trials"], dec["beyond_radius"]))
    rng = np.random.default_rng(7)
    true_pair = exp_affine_pair_test(inst, inst.T_inv[:, 0], inst.T_inv[:, 1])
    random_pair = exp_affine_pair_test(inst, rng.integers(0, inst.p, inst.n), rng.integers(0, inst.p, inst.n))
    print("  exp-affine pair test: the true columns T^-1 e_0, T^-1 e_1 -> constant %s; a random pair -> %d distinct values of m" % (true_pair["constant"], random_pair["distinct_values"]))
    M, q = inst.p, 2**inst.h
    log2_candidates = 2 * inst.n * log2(M)
    log2_constraints = (inst.m - 1) * log2(q)
    print("  key recovery = find two code vectors whose exponentials differ by a constant: M^(2n) candidates (2^%.0f), m-1 field" % log2_candidates)
    print("  equations (2^%.0f), expected spurious solutions 2^%.0f; the n(n-1) true pairs are the needle. No linear handle:" % (log2_constraints, log2_candidates - log2_constraints))
    print("  the condition mixes an additive constraint with exponentials (a discrete-log-with-side-condition problem).")
    if m <= 600:
        sq = _square_rank(np.ascontiguousarray(inst.B_pub.T), inst.p, rng)
        print("  square of im(B_pub): rank %d / %d (the same code as the public form: full)" % (sq["square_rank"], sq["random_expectation"]))
    print("  Key-less controls: im(B T) = im(B) and the centres are unchanged, so the optimization problem is the public one")
    print("  up to x -> T x; script 41's numbers (0.1146-0.1153 vs DQI 0.2518 at m = 6000) apply verbatim.")
    print("  What the key adds: only the key holder can run DQI, so the code carries a signature (section D); the key adds")
    print("  no reach and no new attack on the optimization, only the key-recovery problem above.")


def section_d() -> None:
    print()
    print("=" * 110)
    print("(D) uses: the keyless verifiable challenge, and hash-and-sign on the mixed key (cosine objective, R = 2)")
    print("=" * 110)
    m, alpha = 6000, 0.07
    q_pr = cosine_baseline(alpha)
    q_dqi = general_objective_finite_quality(m, (int(round(alpha * m)) - 1) // 2, 0.0, 1 / sqrt(2), 0.0).value
    print("  design point m = %d, alpha = %.2f: baseline %.3f, key-less portfolio 0.115 (script 41), DQI %.4f" % (m, alpha, q_pr, q_dqi))
    print("  forgery bound for 2^40 information sets at threshold tau: 2^40 exp(-2 (tau - alpha)^2 m / ((1-alpha) R^2)), R = 2:")
    for tau in (0.12, 0.15, 0.18, 0.20, 0.22):
        bound = min(1.0, 2.0**40 * exp(-2 * (tau - q_pr) ** 2 * m / ((1 - alpha) * 4)))
        print("    tau = %.2f: forgery bound %.2e %s" % (tau, bound, "(above the measured portfolio)" if tau > 0.115 else "(below the measured portfolio: not a threshold)"))
    print("  Signing failure needs the DQI law's spread for the cosine objective, which the tridiagonal gives only in mean;")
    print("  the mean 0.252 sits 0.05 above tau = 0.20, where the forgery bound is already below 1e-12. Keyless challenge:")
    print("  sample (t, alpha, centres), ask for x, accept at tau m; soundness rests on one assumption (no classical")
    print("  algorithm reaches DQI's value on the log-Cauchy instance), with no key-indistinguishability; not a proof.")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
