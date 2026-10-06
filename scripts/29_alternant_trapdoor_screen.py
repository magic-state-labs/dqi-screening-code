"""Script 29 - The alternant trapdoor family, and gate D4' (licensed in measure).

(A) Gate D4' and the in-measure cap.  Canonical DQI decodes a uniform superposition over
    the error patterns of each weight, so the decoder only has to succeed on all but a
    vanishing fraction of *uniformly random* weight-<= ell patterns (Jordan et al.
    2408.08292, Sec. 10.3-10.4, Thm 10.1).  Such an in-measure licence is capped by the
    in-measure Singleton bound ``lambda < delta <= alpha`` (for ``ell >= d`` every pattern
    has ``q^(ell-d+1)-1`` same-syndrome competitors of weight <= ell), so at ``rho = 1/2``
    the margin over Prange is at most ``max_alpha sqrt(alpha(1-alpha)) - alpha/2``, whose
    stationary point solves ``5 alpha^2 - 5 alpha + 1 = 0``: ``(sqrt5-1)/4 = 0.309`` at
    ``alpha = (5-sqrt5)/10``, against ``(sqrt2-1)/2 = 0.207`` for the worst-case licence.
    This section checks the closed form against a grid.

(B) The alternant trapdoor family.  ``B = H_pub^T`` where ``H_pub`` is the scrambled
    parity-check matrix of an alternant code over a prime field ``F_p`` with *fixed*
    extension degree ``s`` (a Niederreiter public key), length ``m <= p^s``, accepted sets
    of density 1/2.  The decoder code is the alternant code, decoded by Berlekamp--Massey
    on the hidden GRS to its designed distance: a worst-case licence, so the strict gate
    D4 passes.  With the generic dimension ``m - s r`` the optimizer rate is
    ``alpha = s r/m``, so ``delta = alpha/s`` and ``lambda = alpha/(2s)``.  The classical
    solver sees only ``B``; the key holder sees trace-OPI over ``F_(p^s)``.  This section
    runs gates D0, D2, D3 (at rho=1/2 the same as D2) and D6 (prime alphabet, void), the
    regime factor ``p^alpha rho``, the existential line ``Q*``, and the restart converse
    threshold ``m >= (1-alpha) ln(N/eta) / (2 mu^2)`` at ``N = 2^40``, ``eta = 0.05``,
    against the largest buildable length ``p^s``.

(C) Finite Lemma 9.2 values at three design points, with the finite restart threshold.

(D) Binary Goppa at Classic McEliece parameters: a finite curiosity with a small positive
    margin that dies to best-of-N restarts at these sizes and vanishes asymptotically
    (``lambda = alpha/log2 m``).  Included so the boundary of the family is explicit.

Nothing here is an advantage claim.  The hardness route for (B) is conditional on two
named assumptions (key indistinguishability; Prange-optimality on random dense
max-LINSAT); the finite-length comparison is script 30.  Known structural attacks: the
quadratic-extension key recovery of Couvreur--Otmani--Tillich (1402.3264, wild Goppa,
s=2) and the high-rate distinguisher / key recovery (2306.10294, 2304.14757), which
needs rate loss ``alpha = O(m^(-1/2))``; neither covers fixed ``s >= 3`` at constant
``alpha``.  Rows at ``s = 2`` are marked accordingly.

Run from the repository root:  python scripts/29_alternant_trapdoor_screen.py
"""

from __future__ import annotations

import os
import sys
from math import log, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import (  # noqa: E402
    IN_MEASURE_CAP_RATE,
    IN_MEASURE_MARGIN_CAP,
    alternant_half_designed_reach,
    alternant_reach_window,
    in_measure_margin_cap_at,
    licensed_radius_is_realizable,
    prange_quality,
    qary_entropy,
)
from dqi_explorer.spectral import (  # noqa: E402
    canonical_asymptotic_quality,
    canonical_finite_quality,
)

RESTARTS = 2**40
FAILURE = 0.05
WORST_CASE_CAP = (sqrt(2) - 1) / 2


# --------------------------------------------------------------------------
# helpers (same formulas as 10_advantage_checklist_sweep.py)
# --------------------------------------------------------------------------
def kl_nats(a: float, b: float) -> float:
    if a <= 0:
        return -log(1 - b)
    if a >= 1:
        return -log(b)
    return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))


def existential_quality(alpha: float, rho: float, q: float) -> float:
    budget = alpha * log(q)
    if budget >= -log(rho):
        return 1.0
    lower, upper = rho, 1.0
    for _ in range(200):
        middle = (lower + upper) / 2
        if kl_nats(middle, rho) < budget:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def hoeffding_length(alpha: float, margin: float, draws: int, failure: float) -> float | None:
    """Restart converse: best of ``draws`` information sets scores below Q_DQI*m past this m."""

    if margin <= 0:
        return None
    return (1 - alpha) * log(draws / failure) / (2 * margin**2)


def best_of_n_gain(alpha: float, rho: float, length: int, draws: int) -> float:
    """Approximate gain of the best of ``draws`` information sets over one, as a fraction."""

    sigma = sqrt((1 - alpha) * length * rho * (1 - rho))
    return sigma * sqrt(2 * log(draws)) / length


def margin(alpha: float, reach: float, rho: float = 0.5) -> float:
    return canonical_asymptotic_quality(reach, rho) - prange_quality(alpha, rho)


# --------------------------------------------------------------------------
# (A) gate D4' and the in-measure cap
# --------------------------------------------------------------------------
def section_a() -> None:
    print("=" * 100)
    print("(A) gate D4' (licensed in measure): reach ceiling alpha, cap (sqrt5-1)/4")
    print("=" * 100)
    print("  worst-case licence : lambda <= alpha/2, cap (sqrt2-1)/2 = %.12f at alpha = %.6f" % (WORST_CASE_CAP, 1 - 1 / sqrt(2)))
    print("  in-measure licence : lambda <  alpha,   cap (sqrt5-1)/4 = %.12f at alpha = %.6f" % (IN_MEASURE_MARGIN_CAP, IN_MEASURE_CAP_RATE))
    recomputed = in_measure_margin_cap_at(IN_MEASURE_CAP_RATE)
    print("  recomputed from the quality laws at alpha*: %.12f (agreement %.1e)" % (recomputed, abs(recomputed - IN_MEASURE_MARGIN_CAP)))
    grid = max((in_measure_margin_cap_at(i / 200000), i / 200000) for i in range(1, 200000))
    print("  200000-point grid maximum: %.12f at alpha = %.6f  (stationary point 5a^2-5a+1=0 confirmed)" % grid)
    print()
    print("  what D4' changes and does not change:")
    print("    - D0, D1, D2, D3, D5, D6 are unchanged; D2 is still lambda(1-lambda) > alpha^2/4, now at the in-measure reach.")
    print("    - the reach ceiling doubles (alpha/2 -> alpha) and the cap rises 0.207 -> 0.309.")
    print("    - it admits typical-error decoders: interleaved RS (s/(s+1))alpha, folded RS alpha-eps, RS power decoding to 1-sqrt(1-alpha).")
    print("    - every such decoder is for an RS-derived code; D4' widens the OPI-adjacent zone and manufactures no non-OPI family.")
    print()
    print("  %-8s %-10s %-10s %-10s %-10s" % ("alpha", "cap(wc)", "cap(D4')", "iRS s=4", "power-dec"))
    for alpha in (0.1, 0.2, 0.25, IN_MEASURE_CAP_RATE, 0.3, 0.4, 0.5):
        print(
            "  %-8.4f %-10.4f %-10.4f %-10.4f %-10.4f"
            % (
                alpha,
                margin(alpha, alpha / 2),
                in_measure_margin_cap_at(alpha),
                margin(alpha, 0.8 * alpha),
                margin(alpha, 1 - sqrt(1 - alpha)),
            )
        )
    print("  (iRS s=4: collaborative decoding through (s/(s+1))alpha on random block errors; power-dec: Johnson radius on random errors.)")


# --------------------------------------------------------------------------
# (B) the alternant trapdoor screen
# --------------------------------------------------------------------------
PRIMES = (7, 11, 13, 17, 23, 31)
DEGREES = (2, 3, 4, 5, 6)
ALPHAS = (0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40)


def screen_row(s: int, p: int, alpha: float, rho: float = 0.5) -> dict[str, object]:
    reach = alternant_half_designed_reach(alpha, s)
    d0 = licensed_radius_is_realizable(alpha, reach, p)
    d2 = reach * (1 - reach) > alpha * alpha / 4
    mu = margin(alpha, reach, rho)
    regime = p**alpha * rho
    qstar = existential_quality(alpha, rho, p)
    threshold = hoeffding_length(alpha, mu, RESTARTS, FAILURE)
    length_cap = p**s
    return {
        "s": s,
        "p": p,
        "alpha": alpha,
        "lambda": reach,
        "H_p(lambda)": qary_entropy(reach, p),
        "D0": d0,
        "D2": d2,
        "mu": mu,
        "regime": regime,
        "unsat": regime < 1,
        "Q*": qstar,
        "threshold": threshold,
        "length_cap": length_cap,
        "buildable": threshold is not None and threshold <= length_cap,
        "attack_risk": s == 2,
    }


def section_b() -> list[dict[str, object]]:
    print()
    print("=" * 100)
    print("(B) alternant trapdoor family: B = H_pub^T over F_p, extension degree s, rho = 1/2")
    print("=" * 100)
    print("  D2 window alpha < 2s/(s^2+1):", ", ".join("s=%d: %.4f" % (s, alternant_reach_window(s)) for s in DEGREES))
    print("  D6 is void (prime alphabet); D3 at rho=1/2 coincides with D2; D4 passes (worst-case BM licence with the key).")
    print("  threshold = restart converse (1-alpha) ln(N/eta) / (2 mu^2) at N=2^40, eta=0.05, asymptotic mu; buildable iff threshold <= p^s.")
    print()
    header = "  %-2s %-3s %-6s %-8s %-8s %-3s %-3s %-8s %-7s %-6s %-6s %-9s %-9s %s"
    print(header % ("s", "p", "alpha", "lambda", "H_p(lam)", "D0", "D2", "mu", "p^a/2", "regime", "Q*", "threshold", "p^s", "verdict"))
    print("  " + "-" * 110)
    rows: list[dict[str, object]] = []
    for s in DEGREES:
        for p in PRIMES:
            for alpha in ALPHAS:
                row = screen_row(s, p, alpha)
                rows.append(row)
    shown = 0
    for row in rows:
        if row["p"] not in (11, 13, 31) or row["alpha"] not in (0.08, 0.12, 0.20, 0.30):
            continue
        verdict = []
        if not row["D0"]:
            verdict.append("D0 FAIL")
        if not row["D2"]:
            verdict.append("D2 FAIL")
        if row["D0"] and row["D2"]:
            verdict.append("UNSAT" if row["unsat"] else "sat-regime")
            verdict.append("buildable" if row["buildable"] else "too short")
        if row["attack_risk"]:
            verdict.append("ATTACK RISK (s=2, 1402.3264)")
        print(
            header
            % (
                row["s"],
                row["p"],
                "%.2f" % row["alpha"],
                "%.5f" % row["lambda"],
                "%.4f" % row["H_p(lambda)"],
                "ok" if row["D0"] else "no",
                "ok" if row["D2"] else "no",
                "%+.4f" % row["mu"],
                "%.3f" % row["regime"],
                "UNSAT" if row["unsat"] else "sat",
                "%.3f" % row["Q*"],
                "%.0f" % row["threshold"] if row["threshold"] else "-",
                "%d" % row["length_cap"],
                "; ".join(verdict),
            )
        )
        shown += 1
    print("  (%d of %d screened points shown; the rest are on the same grid.)" % (shown, len(rows)))

    print()
    print("  best point per (s, p): largest mu among points that pass D0, D2, sit in the UNSAT regime and are buildable")
    print("  %-2s %-3s %-6s %-8s %-8s %-9s %-7s %s" % ("s", "p", "alpha", "lambda", "mu", "threshold", "p^s", "note"))
    for s in DEGREES:
        for p in PRIMES:
            eligible = [r for r in rows if r["s"] == s and r["p"] == p and r["D0"] and r["D2"] and r["unsat"] and r["buildable"]]
            if not eligible:
                print("  %-2d %-3d %-6s %-8s %-8s %-9s %-7d none" % (s, p, "-", "-", "-", "-", p**s))
                continue
            best = max(eligible, key=lambda r: r["mu"])
            note = "attack risk" if best["attack_risk"] else ""
            print(
                "  %-2d %-3d %-6.2f %-8.5f %-+8.4f %-9.0f %-7d %s"
                % (s, p, best["alpha"], best["lambda"], best["mu"], best["threshold"], p**s, note)
            )
    return rows


# --------------------------------------------------------------------------
# (C) finite Lemma 9.2 values at design points
# --------------------------------------------------------------------------
DESIGN_POINTS = (
    # (s, p, alpha, m) -- m at or below p^s
    (4, 11, 0.12, 4000),
    (4, 11, 0.12, 14641),
    (3, 13, 0.20, 2197),
    (3, 13, 0.16, 2197),
)


def section_c() -> None:
    print()
    print("=" * 100)
    print("(C) finite Lemma 9.2 values at design points (accepted sets of size (p-1)/2)")
    print("=" * 100)
    print("  %-2s %-3s %-6s %-6s %-5s %-4s %-8s %-8s %-8s %-8s %-9s %s" % ("s", "p", "alpha", "m", "r", "ell", "rho", "Q_fin", "Q_Pr", "mu_fin", "thr(fin)", "clears at m?"))
    for s, p, alpha, m in DESIGN_POINTS:
        # generic alternant: alpha = s r / m, designed distance r + 1, licence 2 ell + 1 < r + 1
        r = round(alpha * m / s)
        ell = (r - 1) // 2
        accepted = (p - 1) // 2
        rho = accepted / p
        quality = canonical_finite_quality(m, p, accepted, ell).value
        baseline = prange_quality(s * r / m, rho)
        mu_fin = quality - baseline
        threshold = hoeffding_length(s * r / m, mu_fin, RESTARTS, FAILURE)
        print(
            "  %-2d %-3d %-6.3f %-6d %-5d %-4d %-8.4f %-8.4f %-8.4f %-+8.4f %-9.0f %s"
            % (s, p, s * r / m, m, r, ell, rho, quality, baseline, mu_fin, threshold if threshold else float("nan"), "yes" if threshold and threshold <= m else "no")
        )
    print("  The finite value sits below the asymptotic one, so the finite thresholds are the ones to use.")


# --------------------------------------------------------------------------
# (D) binary Goppa at Classic McEliece parameters: a finite curiosity
# --------------------------------------------------------------------------
CLASSIC_MCELIECE = (
    # (name, m = code length n, t, k)
    ("mceliece348864", 3488, 64, 2720),
    ("mceliece460896", 4608, 96, 3360),
    ("mceliece6688128", 6688, 128, 5024),
    ("mceliece6960119", 6960, 119, 5413),
    ("mceliece8192128", 8192, 128, 6528),
)


def section_d() -> None:
    print()
    print("=" * 100)
    print("(D) binary Goppa at Classic McEliece parameters (rho = 1/2, singleton sets): finite curiosity")
    print("=" * 100)
    print("  %-16s %-5s %-4s %-7s %-8s %-4s %-8s %-8s %-8s %-8s %s" % ("parameter set", "m", "t", "alpha", "lambda", "D2", "Q_fin", "Q_Pr", "mu_fin", "gain(N)", "verdict"))
    for name, m, t, k in CLASSIC_MCELIECE:
        alpha = (m - k) / m
        reach = t / m
        d2 = reach * (1 - reach) > alpha * alpha / 4
        quality = canonical_finite_quality(m, 2, 1, t).value
        baseline = prange_quality(alpha, 0.5)
        mu_fin = quality - baseline
        gain = best_of_n_gain(alpha, 0.5, m, RESTARTS)
        verdict = "beats one information set" if mu_fin > 0 else "no margin"
        verdict += "; loses to best-of-2^40" if gain > mu_fin else "; survives best-of-2^40"
        print(
            "  %-16s %-5d %-4d %-7.4f %-8.5f %-4s %-8.4f %-8.4f %-+8.4f %-8.4f %s"
            % (name, m, t, alpha, reach, "ok" if d2 else "no", quality, baseline, mu_fin, gain, verdict)
        )
    print("  Asymptotically lambda = alpha/log2(m) -> 0 (sweep row 35), so binary Goppa is the boundary of the family, not a member.")


def section_e() -> None:
    print()
    print("=" * 100)
    print("(E) what this screen says, and what it does not")
    print("=" * 100)
    print("  says:  a prime-field alternant public matrix with fixed s >= 3 passes D0-D6 with a worst-case licence,")
    print("         has margin ~0.06-0.08 at rho=1/2, sits in the unsatisfiable regime at small p, and clears the")
    print("         finite restart threshold inside its own length cap at (s=4, p=11, m=14641); at s=3 the finite")
    print("         threshold (~3000 with sets of size (p-1)/2) needs p >= 17 (p^3 = 4913), not p = 13.")
    print("  says:  to a solver without the key the instance is an unstructured dense p-ary max-LINSAT, not OPI;")
    print("         the hardness route is (A) key indistinguishability + (B) Prange-optimality on random B.")
    print("  does not say: that (A) or (B) holds or that any advantage exists.")
    print("  does not cover: s = 2 (wild Goppa key recovery, 1402.3264) or rate loss alpha = O(m^-1/2)")
    print("         (high-rate distinguisher / key recovery, 2306.10294, 2304.14757).")


# --------------------------------------------------------------------------
# (F) row 28 (interleaved RS) under D4': the in-measure converse made numerical
# --------------------------------------------------------------------------
def ssb_failure_bound(q: int, s: int, N: int, K: int, t: int) -> float:
    """Schmidt--Sidorenko--Bossert (cs/0610074, Thm 6, eq. 17) bound on the failure probability.

    Collaborative decoding of s-interleaved RS[N, K] over F_q with t column errors, the erroneous
    columns uniform on F_q^s minus zero: P_f(t) <= ((q^s - 1/q)/(q^s - 1))^t q^{-(s+1)(t_max - t)}/(q - 1)
    with t_max = s (N - K)/(s + 1) (Thm 2).  Failures are detected (ML-certificate property);
    the miscorrection probability P_e is bounded separately in the paper and is far smaller.
    """

    t_max = s * (N - K) / (s + 1)
    if t > t_max:
        return 1.0
    return ((q**s - 1 / q) / (q**s - 1)) ** t * q ** (-(s + 1) * (t_max - t)) / (q - 1)


def section_f() -> None:
    print()
    print("=" * 100)
    print("(F) row 28, interleaved RS q=401 N=400 K=300, under D4': the converse with an in-measure radius")
    print("=" * 100)
    print("  Theorem 10.1 of Jordan et al. (binary, random target): <s> >= Q^(m) m - eps (m+1), eps = max_k (failure")
    print("  fraction on uniformly random weight-k patterns).  Converse: replace Q^(m) by Q^(m) - eps (m+1)/m in")
    print('  the theorem "DQI against N Prange restarts".  Over F_q this is conditional on the F_q form of Thm 10.1 (stated as expected, not proved)')
    print("  and on the decoder's failure bound holding under the DQI-induced value distribution.")
    print()
    q, N, K, alpha = 401, 400, 300, 0.25
    block_alphabet = None
    print("  %-3s %-6s %-4s %-10s %-8s %-8s %-8s %-8s %-9s" % ("s", "t_max", "ell", "eps(SSB)", "lambda", "Q_fin", "Q_Pr", "mu_eps", "threshold"))
    for s in (2, 4, 8):
        t_max = s * (N - K) / (s + 1)
        for ell in (int(t_max) - 1, int(t_max)):
            eps = ssb_failure_bound(q, s, N, K, ell)
            block_alphabet = q**s
            accepted = (block_alphabet - 1) // 2
            rho = accepted / block_alphabet
            quality = canonical_finite_quality(N, block_alphabet, accepted, ell).value
            q_pr = prange_quality(alpha, rho)
            mu_eps = quality - eps * (N + 1) / N - q_pr
            threshold = hoeffding_length(alpha, mu_eps, RESTARTS, FAILURE)
            print(
                "  %-3d %-6.2f %-4d %-10.2e %-8.4f %-8.4f %-8.4f %-+8.4f %-9.0f%s"
                % (s, t_max, ell, eps, ell / N, quality, q_pr, mu_eps, threshold if threshold else float("nan"),
                   "   <- row 28 at ell = 79" if (s == 4 and ell == 79) else "")
            )
    print("  Q_fin is canonical_finite_quality(400, 401^s, (401^s-1)/2, ell): the finite tridiagonal at the block")
    print("  alphabet.  Here 2 ell + 1 >= d_block = 101, so the value")
    print("  is licensed only in measure (Thm 10.1), never as a worst-case guarantee.  Worst-case cap at alpha=1/4:")
    print("  %.4f; in-measure cap: %.4f." % (margin(alpha, alpha / 2), in_measure_margin_cap_at(alpha)))


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
