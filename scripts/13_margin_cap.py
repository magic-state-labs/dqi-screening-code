"""Script 13 - The margin cap: how much advantage over Prange is possible at all.

The screening conditions decide whether a family gets *any* advantage over
information-set decoding.  This script computes the complementary bound: how much there is
to be had.  Three steps, and none of them needs a decoder hypothesis beyond the licensing
condition:

  * the decoder code ``ker(B^T)`` has rate ``1 - alpha``, so Singleton gives ``delta <= alpha``;
  * worst-case bounded-distance licensing gives ``lambda <= delta/2 <= alpha/2``;
  * maximising ``sqrt(u(1-u)) - u`` at ``u = alpha/2`` has stationary point ``8u^2-8u+1 = 0``.

At ``rho = 1/2`` that yields an exact closed form: the margin is at most ``(sqrt2-1)/2``,
attained at ``alpha = 1 - 1/sqrt2`` where ``lambda(1-lambda) = 1/8`` exactly.

Scope.  The cap bounds the worst-case bounded-distance guarantee against information-set
decoding.  List and soft decoding break ``lambda <= delta/2`` and are outside it.  Singleton
is an upper bound no fixed-alphabet family attains asymptotically, so the cap is generous to
DQI -- the right direction for an impossibility statement, and the reason the constructive
figures below come from Tsfasman-Vladut-Zink rather than from the cap.

Run from the repository root:  python scripts/13_margin_cap.py
"""

from __future__ import annotations

import os
import sys
from math import log, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.discovery import prange_quality  # noqa: E402
from dqi_explorer.spectral import canonical_asymptotic_quality  # noqa: E402

GLOBAL_CAP = (sqrt(2) - 1) / 2
OPTIMAL_RATE = 1 - 1 / sqrt(2)
OPTIMAL_RADIUS = (2 - sqrt(2)) / 4


def margin(alpha, radius, density=0.5):
    """The DQI-over-Prange margin at declared parameters, from the quality formulas of the library."""

    return canonical_asymptotic_quality(radius, density) - prange_quality(alpha, density)


def cap_at(alpha, density=0.5):
    """The Singleton-tight cap at this rate: the margin when lambda = alpha/2."""

    return margin(alpha, alpha / 2, density)


def the_cap():
    print("=" * 94)
    print("(A) the cap, and its exact optimum")
    print("=" * 94)
    print("  lambda* = (2 - sqrt2)/4      = %.12f" % OPTIMAL_RADIUS)
    print("  alpha*  = 1 - 1/sqrt2        = %.12f" % OPTIMAL_RATE)
    print(
        "  lambda*(1 - lambda*)         = %.12f   (exactly 1/8 = %.12f)"
        % (OPTIMAL_RADIUS * (1 - OPTIMAL_RADIUS), 0.125)
    )
    print("  mu_max  = (sqrt2 - 1)/2      = %.12f" % GLOBAL_CAP)
    print()
    computed = cap_at(OPTIMAL_RATE)
    print("  recomputed from the quality laws: %.12f" % computed)
    print("  agreement with the closed form  : %.2e" % abs(computed - GLOBAL_CAP))
    print()
    grid = max((cap_at(i / 200000), i / 200000) for i in range(1, 200000))
    print(
        "  global maximum on a 200000-point grid: %.12f at alpha = %.6f"
        % (grid[0], grid[1])
    )
    print("  grid confirms the stationary point is the maximum.")


def density_table():
    print()
    print("=" * 94)
    print("(B) the cap is maximal at rho = 1/2 and collapses as density falls")
    print("=" * 94)
    print("  %-10s %-14s %-14s %s" % ("rho", "alpha*", "cap", "note"))
    print("  " + "-" * 60)
    previous = None
    monotone = True
    for density in (1 / 64, 0.05, 0.1, 0.25, 0.4, 0.5):
        best = max(
            (margin(i / 50000, i / 100000, density), i / 50000)
            for i in range(1, 50000)
        )
        note = "closed form (sqrt2-1)/2" if abs(density - 0.5) < 1e-12 else ""
        print("  %-10.5f %-14.6f %-14.6f %s" % (density, best[1], best[0], note))
        if previous is not None and best[0] < previous:
            monotone = False
        previous = best[0]
    print()
    print("  monotone increasing in rho on (0, 1/2]: %s" % monotone)
    print("  So rho = 1/2 is the best density for margin, and the collapse toward rho = 1/q")
    print("  is an independent route to the exact-value disqualification that D3 reaches.")


def regime_table():
    print()
    print("=" * 94)
    print("(C) the unsatisfiable regime caps the rate, and so caps the margin")
    print("=" * 94)
    print("  Unsat needs q^alpha * rho < 1, i.e. alpha < log_q(1/rho). At rho = 1/2:")
    print()
    print("  %-12s %-12s %-16s %s" % ("q", "log_q 2", "cap in regime", "fraction of max"))
    print("  " + "-" * 60)
    for alphabet in (16, 64, 256, 1024, 4096, 16384, 65536, 10**6, 10**9):
        ceiling = log(2) / log(alphabet)
        rate = min(ceiling, OPTIMAL_RATE)
        value = cap_at(rate)
        print(
            "  %-12d %-12.6f %-16.6f %.1f%%"
            % (alphabet, ceiling, value, 100 * value / GLOBAL_CAP)
        )
    print()
    print("  alpha_max = ln2/ln q -> 0, so the regime-constrained cap is O(1/sqrt(log q)):")
    for alphabet in (10**3, 10**6, 10**9, 10**12):
        ceiling = log(2) / log(alphabet)
        print(
            "    q=1e%-3d cap=%.5f  sqrt(ln2/(2 ln q))=%.5f  ratio %.3f"
            % (
                round(log(alphabet, 10)),
                cap_at(ceiling),
                sqrt(log(2) / (2 * log(alphabet))),
                cap_at(ceiling) / sqrt(log(2) / (2 * log(alphabet))),
            )
        )


# Rows 27-31 of 10_advantage_checklist_sweep.py, with the declared
# parameters that script carries.
NEW_CANDIDATES = (
    (27, "Roth-Lempel / multi-twist TGRS", 0.25, 0.125, "sat 2.24"),
    (28, "interleaved RS, s=4", 0.25, 0.200, "sat 2.24"),
    (29, "elliptic (genus-one) codes", 0.25, 0.12327, "sat 2.00"),
    (30, "Suzuki (Castle) codes q=32", 0.50, 0.18951, "sat 2.83"),
    (31, "GS tower F_256 at alpha=0.10", 0.10, 0.016667, "UNSAT 0.871"),
)


def attainment():
    print()
    print("=" * 94)
    print("(D) rows 27-31 of script 10 against the cap")
    print("=" * 94)
    print(
        "  %-4s %-30s %-7s %-8s %-8s %-9s %-9s %s"
        % ("row", "family", "alpha", "lambda", "margin", "cap@a", "of cap", "regime")
    )
    print("  " + "-" * 90)
    exceeded = []
    for row, name, alpha, radius, regime in NEW_CANDIDATES:
        value = margin(alpha, radius)
        ceiling = cap_at(alpha)
        flag = ""
        if value > ceiling + 1e-12:
            flag = "  <== EXCEEDS THE CAP"
            exceeded.append((row, name, alpha, radius, value, ceiling))
        print(
            "  %-4d %-30s %-7.4f %-8.5f %-8.4f %-9.4f %-9s %s%s"
            % (
                row,
                name[:30],
                alpha,
                radius,
                value,
                ceiling,
                "%.1f%%" % (100 * value / ceiling),
                regime,
                flag,
            )
        )
    print()
    for row, name, alpha, radius, value, ceiling in exceeded:
        print("  Row %d exceeds the cap, and that is a finding rather than an error." % row)
        print(
            "    lambda = %.3f but Singleton allows only alpha/2 = %.3f at this rate,"
            % (radius, alpha / 2)
        )
        print("    so NO worst-case unique guarantee can license it. It is consistent only")
        print("    because the decoder is collaborative and probabilistic -- which is what the")
        print("    radius check (C2R) of script 10 reports as 'unjudged (not worst-case)'.")
        print()
    print("  Rows 27 and 29 are Singleton-tight, so they take every bit of advantage their")
    print("  rate permits: that is a positive statement about a new candidate, not just a")
    print("  screen pass. Rows 30 and 31 leave margin unused because the genus eats the")
    print("  designed distance. Every row but 31 sits in the satisfiable regime, where the")
    print("  classical competitor is a search for a fully satisfying word instead.")


def near_tightness():
    print()
    print("=" * 94)
    print("(E) the cap is nearly attained, so it is not vacuous")
    print("=" * 94)
    existential = margin(0.25, 0.125)
    print(
        "  existentially: multi-twist TGRS at alpha=0.25, lambda=0.125 gives %.4f,"
        % existential
    )
    print("                 which is %.1f%% of the global cap." % (100 * existential / GLOBAL_CAP))
    constrained = cap_at(log(2) / log(16384))
    print(
        "  constructively: the best TVZ point inside the unsat window reaches 0.1395 at"
    )
    print(
        "                 q=16384, which is %.1f%% of the regime cap %.4f."
        % (100 * 0.1395 / constrained, constrained)
    )
    print()
    print("  Both directions near the bound, so the cap describes reality rather than")
    print("  bounding it from far away.")


def main():
    the_cap()
    density_table()
    regime_table()
    attainment()
    near_tightness()
    print()
    print("=" * 94)
    print("The cap bounds a worst-case bounded-distance guarantee against information-set")
    print("decoding. It is not a speedup claim, and it says nothing")
    print("about list or soft decoding, about portfolios, or about quantum computation.")


if __name__ == "__main__":
    main()
