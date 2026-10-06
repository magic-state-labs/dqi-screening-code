"""Script 38 - Other settings examined beside the hidden-key candidates, with numbers.

(A) Private-key (symmetric) setting: B itself secret.  The instance-level comparison degenerates
    (a solver without B scores rho); as a scheme it is a MAC or designated-verifier signature
    whose outsider faces an LPN-type learning problem from (sets, x) pairs.  The numbers: how
    many signatures determine B row by row information-theoretically, and the noise rate of the
    induced LPN instance.
(B) Hidden LDPC key with a better-than-BP decoder.  The reach-versus-exposure trade-off of
    script 31 (A5) with the reach as a free parameter: what reach a quantum decoder would
    need at each row weight for the key to clear D2 while staying hidden from information
    sets.
(C) Planted / satisfiable schemes: DQI samples, it does not search; the mass on a planted
    solution is P(m)^2 / Z.
(D) Summary table.

Run from the repository root:  python scripts/38_other_settings.py
"""

from __future__ import annotations

import os
import sys
from math import comb, log, log2

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.dqi_signature import agreement_distribution  # noqa: E402
from dqi_explorer.discovery import prange_quality  # noqa: E402


def h2(x: float) -> float:
    return 0.0 if x <= 0 or x >= 1 else -x * log2(x) - (1 - x) * log2(1 - x)


def section_a() -> None:
    print("=" * 100)
    print("(A) private-key setting: the matrix is secret, the adversary sees (sets, signature) pairs")
    print("=" * 100)
    m, n, p, tau = 6000, 720, 11, 0.57
    rho = 5 / 11
    bits_per_row = n * log2(p)
    # each signature reveals, per row i, one noisy bit: whether b_i . x lies in F_i (true w.p. tau)
    capacity = 1 - h2(tau)  # bits per observation about the row, at best
    needed = bits_per_row / capacity
    print("  design point: m=%d n=%d p=%d, tau=%.2f.  A row of B carries %.0f bits; each signature gives one" % (m, n, p, tau, bits_per_row))
    print("  set-membership observation per row, correct with probability tau, at most %.3f bits, so at least %.0f" % (capacity, needed))
    print("  signatures are needed information-theoretically to pin a row.  Recovering b_i from (x_j, [b_i . x_j in F_j,i])")
    print("  is learning a vector over F_%d from set-membership samples with noise rate %.2f: LPN-type, no polynomial" % (p, 1 - tau))
    print("  algorithm known at this noise; the verifier, who holds B, faces the class-2 problem instead.")
    print("  Not a catalog row: the instance-level comparison (solver without B scores rho = %.3f) is degenerate." % rho)


def section_b() -> None:
    print()
    print("=" * 100)
    print("(B) hidden LDPC key: reach a decoder would need versus exposure by information sets")
    print("=" * 100)
    print("  a planted parity check of weight w is a weight-w word of the public optimizer code; one information set")
    print("  exposes it with probability about w alpha (1-alpha)^(w-1).  D2 at rho = 1/2 needs lambda(1-lambda) > alpha^2/4.")
    print("  %-4s %-6s %-12s %-14s %-16s" % ("w", "alpha", "D2 needs lam", "sets to expose", "BP reach (31 A5)"))
    bp = {6: 0.084, 16: 0.0679, 100: 0.0004}
    for w, alpha in ((6, 0.5), (16, 0.25), (25, 0.2), (50, 0.1), (100, 0.05)):
        need = alpha * alpha / 4
        lam = 0.5 * (1 - (1 - 4 * need) ** 0.5) if need < 0.25 else 0.5
        expose = w * alpha * (1 - alpha) ** (w - 1)
        print("  %-4d %-6.2f %-12.4f %-14.0f %-16s" % (w, alpha, lam, 1 / expose, ("%.4f" % bp[w]) if w in bp else "-"))
    print("  reading: for w >= 25 the key hides only from about 40 or more information sets at alpha <= 0.2, and the")
    print("  reach a decoder would need (0.01-0.03) is what BP gives only at w <= 16.  A quantum decoder with reach above")
    print("  BP's changes the row-weight window; it does not remove the exposure column, which is decoder-independent.")
    print("  This is the same trade-off as script 31 (A5); the locally-quantum decoder of 2604.24633 is not rerun here.")


def section_c() -> None:
    print()
    print("=" * 100)
    print("(C) planted / satisfiable schemes: DQI does not find a planted perfect solution")
    print("=" * 100)
    m, p, r, ell = 6000, 11, 5, 89
    dist = agreement_distribution(m, p, r, ell)
    P = dist["polynomial"]
    log_Z = np.log((dist["probabilities"] * 1.0).sum())  # normalized, so use the model normalizer explicitly
    rho = r / p
    from math import lgamma

    logs = np.array([lgamma(m + 1) - lgamma(f + 1) - lgamma(m - f + 1) + f * log(rho) + (m - f) * log(1 - rho) for f in range(m + 1)])
    logZ = np.logaddexp.reduce(logs + 2 * np.log(np.abs(P) + 1e-300)) + 720 * log(p)  # Z = p^n E[P^2]
    mass = 2 * log(abs(P[m])) - logZ
    print("  design point m=%d: log10 Pr[planted x* with f = m] = %.1f (P(m)^2 / Z); the encryption-style" % (m, mass / log(10)))
    print("  formulations (list-recovery McEliece, planted max-LINSAT) ask for that x*, which DQI's sample never returns.")


def section_d() -> None:
    print()
    print("=" * 100)
    print("(D) summary table")
    print("=" * 100)
    rows = (
        ("K1 sparse-mixed GRS key, w=3", "secret sparse mixing (not a subfield)", "alpha/6", "+0.072 (finite +0.064)", "passes D0-D6; square battery full; outside the 2014 attack regime", "passes the screen; no finite run in this repository"),
        ("K1 sparse-mixed GRS key, w=2", "same", "alpha/4", "+0.096", "dim(square of im(B)) <= 2r-1+m/2 < m for alpha < 1/4", "closed in the UNSAT regime"),
        ("K2 column-inserted GRS key, u=r", "secret inserted functionals + pair mixing", "alpha/4", "+0.096 (finite +0.090)", "passes D0-D6; battery full; boundary of the 2019 attack (unverified)", "passes the screen; no finite run in this repository"),
        ("K3 expanded GRS key", "secret field bases per position", "alpha/(2s)", "+0.062", "Couvreur-Lequesne 2022 distinguisher (subspace dim > s/2)", "closed"),
        ("S1 hash-and-sign, quantum signer", "scheme on any class-2 key", "-", "tau in (portfolio, Q_DQI)", "the law of the signature does not depend on the key", "scheme"),
        ("C1 classical key holder", "baseline with the key", "-", "-", "F_q-line moves pin r-1 rows against n-1 for F_p-lines", "script 36: below the key-less portfolio"),
        ("private-key setting", "B secret", "-", "-", "LPN-type learning from signatures; not a catalog row", "recorded"),
        ("hidden LDPC + better decoder", "sparse secret", "-", "-", "exposure column is decoder-independent", "recorded"),
        ("planted schemes", "satisfiable regime", "-", "-", "DQI samples, mass on x* is P(m)^2/Z", "recorded"),
    )
    print("  %-34s %-40s %-10s %-24s %-66s %s" % ("problem", "mechanism", "reach", "margin", "screen", "status"))
    for row in rows:
        print("  %-34s %-40s %-10s %-24s %-66s %s" % row)


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
