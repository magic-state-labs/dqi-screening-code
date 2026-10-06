"""Script 48 - The numbers the paper quotes, recomputed and printed beside the paper's value.

Closed-form values are recomputed from their definitions.  Values that come from long runs
(the classical solvers, the lattice attacks, the cost estimates) are read from the stored
records under ``results/``.  Each line shows the computed value, the value printed in the
paper and ``ok`` or ``DIFFERS``.

  (1) the reach threshold and the bound on the margin (Theorem 6, Proposition 8)
  (2) Table 2: finite-length comparisons with classical algorithms
  (3) Table 3 and the alternant (McEliece) parameters
  (4) alternant max-agreement: key holder, list recovery, attack-cost estimates
  (5) multiplicative polynomial intersection: existential line, code structure
  (6) Table 5: row-subset lattice attacks
  (7) rule (R4): subfield examples
  (8) solver budgets (Table 8)

Run from the repository root:
  python scripts/48_paper_derived_numbers.py
  python scripts/48_paper_derived_numbers.py --witnesses   # also re-score the stored assignments (about an hour)

The record goes to ``results/paper-derived-numbers.json``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from math import comb, log, sqrt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.spectral import canonical_finite_quality  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "paper-derived-numbers.json")
record: dict = {}
differs: list[str] = []


# ---------------------------------------------------------------------------------------
# formulas of the paper
# ---------------------------------------------------------------------------------------
def q_dqi(lam: float, rho: float) -> float:
    """Asymptotic DQI quality for the indicator objective (the semicircle law)."""
    if lam > 1 - rho:
        return 1.0
    return (sqrt(lam * (1 - rho)) + sqrt(rho * (1 - lam))) ** 2


def q_pr(alpha: float, rho: float) -> float:
    """One Prange completion."""
    return rho + (1 - rho) * alpha


def lambda_crit(alpha: float, rho: float) -> float:
    b = q_pr(alpha, rho)
    return (sqrt((1 - rho) * b) - sqrt(rho * (1 - b))) ** 2


def rho_crit(alpha: float) -> float:
    return 0.5 * (1 - sqrt(2 * (1 - alpha) / (2 - alpha)))


def cosine_finite(m: int, ell: int) -> float:
    """Exact finite DQI mean cosine: sigma * lambda_max(A) / m with sigma = 1/sqrt2, zero skewness."""
    A = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        A[k - 1, k] = A[k, k - 1] = sqrt(k * (m - k + 1))
    return float(np.linalg.eigvalsh(A)[-1]) / (sqrt(2) * m)


def restart_threshold(alpha: float, margin: float, R: float) -> float:
    """Theorem "DQI against N Prange restarts" at N = 2^40, eta = 0.05."""
    return (1 - alpha) * R * R * log(2 ** 40 / 0.05) / (2 * margin * margin)


def kl(a: float, b: float) -> float:
    return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))


def qary_entropy(x: float, q: int) -> float:
    return (x * log(q - 1) - x * log(x) - (1 - x) * log(1 - x)) / log(q)


def bisect(f, lo: float, hi: float, steps: int = 200) -> float:
    flo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bessel(nu: int, x: float, terms: int = 80) -> float:
    total, term = 0.0, (x / 2) ** nu / math.factorial(nu)
    for k in range(terms):
        total += term
        term *= (x / 2) ** 2 / ((k + 1) * (k + 1 + nu))
    return total


def cosine_qstar(alpha: float, q: int) -> tuple[float, float]:
    """Existential line for the cosine: tilt beta with KL = alpha ln q, and the mean I_1/I_0."""
    target = alpha * log(q)
    beta = bisect(lambda b: b * bessel(1, b) / bessel(0, b) - log(bessel(0, b)) - target, 1e-6, 60.0)
    return beta, bessel(1, beta) / bessel(0, beta)


# ---------------------------------------------------------------------------------------
# printing and reading
# ---------------------------------------------------------------------------------------
def show(key: str, value, paper, digits: int | None = None, ref: str = "paper") -> None:
    """Print ``value`` beside the paper's ``paper``; with ``digits`` the value is rounded first."""
    if digits is None:
        ok = value == paper
    else:
        ok = abs(round(float(value), digits) - float(paper)) < 1e-9
    record[key] = {"value": value, "paper": paper}
    shown = value if isinstance(value, (str, int)) else "%.6g" % value
    print("  %-70s %-22s %s: %-20s %s" % (key, shown, ref, paper, "ok" if ok else "DIFFERS"))
    if not ok:
        differs.append(key)


def span(values, fmt: str = "%.4f") -> str:
    lo, hi = fmt % min(values), fmt % max(values)
    return lo if lo == hi else "%s to %s" % (lo, hi)


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def draft(name: str) -> dict:
    return load(name)["results"][0]


def alternant(m: int, seed: int, tag: str = "ext6bud600") -> dict:
    return draft("finite-control-alternant-run-%s-m%d-s%d.json" % (tag, m, seed))


def multiplicative_run(m: int, seed: int, tag: str = "bud600-") -> dict:
    return draft("finite-control-screen-logcauchy-run-%sm%d-s%d.json" % (tag, m, seed))


def opi_runs(m: int) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "opi-finite-control-run-s*.json"))):
        with open(path, encoding="utf-8") as fh:
            out += [r for r in json.load(fh)["results"] if r["checks"]["m"] == m]
    return out


def lattice_records(name: str) -> list[dict]:
    return load(os.path.join("lattice", name + ".lattice.json"))["records"]


SEEDS = (3101, 3102, 3103)


# (1) the reach threshold and the bound on the margin -----------------------------------
def rules() -> None:
    print("(1) reach threshold (Theorem 'reach threshold'), rule (R1)(c) and the margin bound kappa")
    rng = np.random.default_rng(48)
    bad_threshold = bad_density = 0
    for _ in range(200000):
        a, r, lam = rng.random(3)
        bad_threshold += (q_dqi(lam, r) > q_pr(a, r)) != (lam > lambda_crit(a, r))
        bad_density += (a / 2 > lambda_crit(a, r) + 1e-15) != (r > rho_crit(a))
    show("random (alpha, rho, lambda) violating Q_DQI > Q_Pr <=> lambda > lambda_crit", int(bad_threshold), 0)
    show("random (alpha, rho) violating alpha/2 > lambda_crit <=> rho > rho_crit", int(bad_density), 0)
    show("lambda_crit(0.3, 1/2) - (1 - sqrt(1 - 0.3^2))/2", lambda_crit(0.3, 0.5) - (1 - sqrt(1 - 0.09)) / 2, 0, 12)
    u = (1 - 1 / sqrt(2)) / 2
    show("kappa at alpha = 1 - 1/sqrt2, minus (sqrt2 - 1)/2", sqrt(u * (1 - u)) - u - (sqrt(2) - 1) / 2, 0, 12)


# (2) Table 2 ---------------------------------------------------------------
def experiments() -> None:
    print("(2) Table 2: Q_Pr, best classical (range over seeds), exact finite DQI, classical - DQI")

    def row(label, qpr, best, dqi, paper):
        show(label + ": Q_Pr", qpr, paper[0], 3)
        show(label + ": best classical", span(best), paper[1])
        show(label + ": DQI (finite)", dqi, paper[2], 4)
        show(label + ": classical - DQI", span([b - dqi for b in best], "%+.3f"), paper[3])

    for m, paper in ((400, (0.549, "0.6700 to 0.6725", 0.6856, "-0.016 to -0.013")),
                     (4000, (0.550, "0.5850 to 0.5900", 0.7109, "-0.126 to -0.121"))):
        runs = opi_runs(m)
        c = runs[0]["checks"]
        dqi = canonical_finite_quality(m, c["p"], c["r"], c["ell"]).value
        row("OPI, m = %d" % m, c["Q_Pr"], [r["best_classical"] / m for r in runs], dqi, paper)
    for m, seeds, paper in ((14000, SEEDS, (0.498, "0.5144 to 0.5171", 0.5321, "-0.018 to -0.015")),
                            (20000, SEEDS, (0.498, "0.5092 to 0.5128", 0.5329, "-0.024 to -0.020"))):
        runs = [alternant(m, s) for s in seeds]
        dqi = canonical_finite_quality(m, 11, 5, runs[0]["ell"]).value
        row("AMA (s = 6), m = %d" % m, runs[0]["Q_Pr"], [r["best_classical"] / m for r in runs], dqi, paper)
    best = []
    for seed in SEEDS:
        recs = lattice_records("logcauchy-m600-s%d" % seed)
        best.append(max(r["best_all_rows"] for r in recs if r["block"] == 20 and r["rows"] in (100, 150)))
    row("MPI, lattice, m = 600", 42 / 600, best, cosine_finite(600, 20), (0.070, "0.2569 to 0.2873", 0.2235, "+0.033 to +0.064"))
    runs = [multiplicative_run(2000, s) for s in SEEDS]
    row("MPI, m = 2000", 0.07, [r["best_classical"] / 2000 for r in runs], cosine_finite(2000, 69),
        (0.070, "0.1493 to 0.1521", 0.2433, "-0.094 to -0.091"))
    best = [max(multiplicative_run(6000, s)["best_classical"], multiplicative_run(6000, s, "")["best_classical"]) / 6000 for s in SEEDS]
    row("MPI, m = 6000 (better of the 40 s and 600 s runs)", 0.07, best, cosine_finite(6000, 209),
        (0.070, "0.1146 to 0.1153", 0.2518, "-0.137"))

    print("    budget sensitivity (seed 3101, line search and annealing budgets 600 s -> 1800 s)")
    b6, b18 = alternant(14000, 3101), alternant(14000, 3101, "ext6bud1800")
    dqi = b6["Q_fin"]["value"]
    show("AMA m = 14000: best at 600 s", b6["best_classical"] / 14000, 0.5144, 4)
    show("AMA m = 14000: best at 1800 s", b18["best_classical"] / 14000, 0.5157, 4)
    show("AMA m = 14000: gain in rows", b18["best_classical"] - b6["best_classical"], 19)
    show("AMA m = 14000: gain in quality", (b18["best_classical"] - b6["best_classical"]) / 14000, 0.0014, 4)
    show("AMA m = 14000: gap to DQI at 600 s", dqi - b6["best_classical"] / 14000, 0.018, 3)
    show("AMA m = 14000: gap to DQI at 1800 s", dqi - b18["best_classical"] / 14000, 0.016, 3)
    c6, c18 = multiplicative_run(6000, 3101), multiplicative_run(6000, 3101, "bud1800-")
    dqi = c6["Q_fin"]["value"]
    show("MPI m = 6000: best at 600 s", c6["best_classical"] / 6000, 0.1146, 4)
    show("MPI m = 6000: best at 1800 s", c18["best_classical"] / 6000, 0.1169, 4)
    show("MPI m = 6000: gain in mean cosine", (c18["best_classical"] - c6["best_classical"]) / 6000, 0.0023, 4)
    show("MPI m = 6000: gap to DQI at 600 s", dqi - c6["best_classical"] / 6000, 0.137, 3)
    show("MPI m = 6000: gap to DQI at 1800 s", dqi - c18["best_classical"] / 6000, 0.135, 3)

    print("    smaller OPI lengths (best of line search and annealing per seed)")
    for m, paper_best, paper_dqi in ((100, "0.79 to 0.80", 0.637), (210, "0.729 to 0.733", 0.674)):
        runs = opi_runs(m)
        best = [max(r["attacks"]["B_line"]["fraction"], r["attacks"]["C_heat"]["fraction"]) for r in runs]
        show("OPI m = %d: line search and annealing" % m, span(best, "%.2f" if m == 100 else "%.3f"), paper_best)
        show("OPI m = %d: DQI (finite)" % m, runs[0]["checks"]["dqi_lemma92"] / m, paper_dqi, 3)


# (3) Table 3 and the alternant parameters ------------------------------
def candidate_rules() -> None:
    print("(3) Table 3: alternant max-agreement (AMA) column (p = 11, s = 6, alpha = 0.08, rho = 5/11, m = 14000)")
    p, s, alpha, rho = 11, 6, 0.08, 5 / 11
    r14, r20 = alternant(14000, 3101), alternant(20000, 3101)
    show("optimal rate alpha* = s(1 - s/sqrt(1+s^2))", s * (1 - s / sqrt(1 + s * s)), 0.0816, 4)
    show("parity rows r", r14["r"], 187)
    show("columns n = s r", r14["n"], 1122)
    show("rank of the public matrix", r14["rank"], 1122)
    show("licensed radius ell = floor((r-1)/2)", r14["ell"], 93)
    show("designed distance d >= r + 1", r14["designed_distance"], 188)
    show("(R1)(a): delta >= alpha/s at the instance rate", r14["alpha"] / s, 0.0134, 4)
    show("(R1)(a): 2 lambda_crit", 2 * lambda_crit(alpha, rho), 0.0038, 4)
    show("(R1)(b): H_11^{-1}(alpha)", bisect(lambda x: qary_entropy(x, p) - alpha, 1e-12, 1 - 1 / p), 0.028, 3)
    show("lambda_crit(alpha, rho)", lambda_crit(alpha, rho), 0.0019, 4)
    show("(R1)(c): rho_crit(alpha)", rho_crit(alpha), 0.0105, 4)
    show("(R1)(c): Plotkin reach (10/11) alpha/2", (1 - 1 / p) * alpha / 2, 0.036, 3)
    lam = alpha / (2 * s)
    show("margin at rho = 1/2", sqrt(lam * (1 - lam)) - alpha / 2, 0.041, 3)
    show("margin at rho = 5/11", q_dqi(lam, rho) - q_pr(alpha, rho), 0.038, 3)
    mu14 = canonical_finite_quality(14000, p, 5, 93).value - r14["Q_Pr"]
    show("exact margin at m = 14000", mu14, 0.034, 3)
    show("Q* at alpha = 0.08, rho = 5/11", bisect(lambda Q: kl(Q, rho) - alpha * log(p), rho + 1e-12, 1 - 1e-12), 0.76, 2)
    show("regime factor p^alpha rho", p ** alpha * rho, 0.55, 2)
    show("Prange-restart threshold at m = 14000", math.ceil(restart_threshold(r14["alpha"], mu14, 1)), 12373)
    show("largest length p^s", p ** s, 1771561)
    show("m = 20000: ell", r20["ell"], 133)
    show("m = 20000: designed distance", r20["designed_distance"], 268)
    mu20 = canonical_finite_quality(20000, p, 5, 133).value - r20["Q_Pr"]
    show("m = 20000: exact margin", mu20, 0.035, 3)
    show("m = 20000: Prange-restart threshold", math.ceil(restart_threshold(r20["alpha"], mu20, 1)), 11770)
    show("Q_DQI - 0.01 at m = 14000", r14["Q_fin"]["value"] - 0.01, 0.522, 3)
    # the hardness assumptions are stated at m = 40000: r = round(alpha m / s) = 533, ell = floor((r - 1)/2) = 266
    q40 = canonical_finite_quality(40000, p, 5, 266).value
    show("m = 40000: parity rows r = round(0.08 m / 6)", round(0.08 * 40000 / 6), 533)
    show("m = 40000: Q_DQI (finite)", q40, 0.534, 3)
    show("Assumption (B): Q_DQI - 0.01 at m = 40000", q40 - 0.01, 0.524, 3)
    alpha40 = 6 * 533 / 40000
    mu40 = q40 - (rho + (1 - rho) * alpha40)
    show("m = 40000: ell", (533 - 1) // 2, 266)
    show("m = 40000: designed distance r + 1", 533 + 1, 534)
    show("m = 40000: exact margin", mu40, 0.036, 3)
    show("m = 40000: Prange-restart threshold", math.ceil(restart_threshold(alpha40, mu40, 1)), 10991)

    print("    multiplicative polynomial intersection (MPI) column (M = 8191, alpha = 0.07, cosine objective)")
    alpha, M = 0.07, 8191
    show("limiting reach alpha/2", alpha / 2, 0.035, 3)
    show("cosine threshold (1 - sqrt(1 - 2 alpha^2))/2", (1 - sqrt(1 - 2 * alpha ** 2)) / 2, 0.0025, 4)
    h2 = lambda x: -x * log(x, 2) - (1 - x) * log(1 - x, 2)
    show("(R1)(b): largest lambda with H_2(lambda) + lambda <= alpha log2 M", bisect(lambda x: h2(x) + x - alpha * log(M, 2), 1e-12, 0.5), 0.196, 3)
    show("Q* for the cosine", cosine_qstar(alpha, M)[1], 0.726, 3)
    show("asymptotic margin sqrt(2 lambda (1 - lambda)) - alpha", sqrt(2 * (alpha / 2) * (1 - alpha / 2)) - alpha, 0.19, 2)
    mu = cosine_finite(6000, 209) - alpha
    show("exact margin at m = 6000", mu, 0.18, 2)
    show("Prange-restart threshold at m = 6000", math.ceil(restart_threshold(alpha, mu, 2)), 1729)
    show("Assumption (C): Q_DQI - 0.01 at m = 6000", cosine_finite(6000, 209) - 0.01, 0.242, 3)


# (4) alternant max-agreement -----------------------------------------------------------
def alternant_attacks() -> None:
    print("(4) alternant max-agreement (AMA): key holder, list recovery, attack-cost estimates")
    show("list size L = (5/11) 11^6", 5 * 11 ** 5, 805255)
    show("list-recovery threshold sqrt(186 * 14000 * L) / 1e6", sqrt(186 * 14000 * 5 * 11 ** 5) / 1e6, 1.4, 1)
    starts, bests, moves, decoded = [], [], [], []
    for seed in SEEDS:
        k = draft("finite-control-alternant-keyholder-run-ext6warm600-m14000-s%d.json" % seed)
        starts.append(k["keyless_best"])
        bests.append(max(k["attacks"][a]["best"] for a in ("KE_line_q_warm", "KE_heat_q_warm")) / 14000)
        moves += [k["attacks"][a]["moves"] for a in ("KE_line_q_warm", "KE_heat_q_warm")]
        d = alternant(14000, seed)["decoder"]
        decoded.append("%d/%d" % (d["exact_recoveries"], d["trials"]))
    show("key holder: starting qualities (seeds 3101-3103)", ", ".join("%.4f" % v for v in starts), "0.5144, 0.5151, 0.5171")
    show("key holder: best after 600 s of line search and of annealing", ", ".join("%.4f" % v for v in bests), "0.5144, 0.5151, 0.5171")
    show("key holder: moves per run, rounded to tens", span([round(v, -1) for v in moves], "%d"), "180")
    show("decoder: random patterns recovered within ell = 93", ", ".join(decoded), "50/50, 50/50, 50/50")
    sets = load("gijs-cost-estimate.json")
    ours, larger = sets["alternant_sets"]["ours s=6 n=14000 r=187"], sets["alternant_sets"]["s=6 n=20000 r=267"]
    show("distinguisher, theorem-backed variant, m = 14000: log2 cost", ours["min_rigorous"]["log2_T_bits"], 123, 0)
    show("distinguisher, heuristic variant, m = 14000: log2 cost", ours["min_incl_heuristic"]["log2_T_bits"], 106, 0)
    show("distinguisher, theorem-backed variant, m = 20000: log2 cost", larger["min_rigorous"]["log2_T_bits"], 127, 0)
    show("distinguisher, heuristic variant, m = 20000: log2 cost", larger["min_incl_heuristic"]["log2_T_bits"], 110, 0)
    show("key recovery, m = 14000: log2 cost", sets["key_recovery_gijs_sec7"]["ours s=6 n=14000 r=187"]["log2_T_rec_bits"], 158, 0)
    show("key recovery, m = 40000: log2 cost", sets["key_recovery_gijs_sec7"]["s=6 n=40000 r=533"]["log2_T_rec_bits"], 159, 0)
    longer = sets["alternant_sets"]["s=6 n=40000 r=533"]
    show("distinguisher, theorem-backed variant, m = 40000: log2 cost", longer["min_rigorous"]["log2_T_bits"], 121, 0)
    show("distinguisher, heuristic variant, m = 40000: log2 cost", longer["min_incl_heuristic"]["log2_T_bits"], 117, 0)


# (5) multiplicative polynomial intersection --------------------------------------------
def multiplicative() -> None:
    print("(5) multiplicative polynomial intersection (MPI): existential line and code structure")
    alpha, M = 0.07, 8191
    beta, qstar = cosine_qstar(alpha, M)
    show("entropy budget alpha ln q", alpha * log(M), 0.6308, 4)
    show("tilt", beta, 2.186, 3)
    show("Q* at q = 8191", qstar, 0.726, 3)
    show("Q* at q = 2^17 - 1", cosine_qstar(alpha, 2 ** 17 - 1)[1], 0.803, 3)
    show("Q* at q = 2^31 - 1", cosine_qstar(alpha, 2 ** 31 - 1)[1], 0.945, 3)
    show("m = 6000, n = 420: decoder-code dimension", 6000 - 420, 5580)
    show("m = 6000, n = 420: ell", (420 - 1) // 2, 209)
    show("m = 6000, n = 420: distance a Hamming license needs (2 ell + 2)", 2 * 209 + 2, 420)
    show("m = 6000, n = 420: Singleton bound on the distance", 6000 - 5580 + 1, 421)
    probe = load("decoder-probe-logcauchy-s40.json")["results"][0]
    show("Schur square of the optimizer code at m = 600, n = 60: dimension", probe["schur_square"]["square_rank"], 600)
    show("Schur square of a Reed-Solomon code of dimension 60: 2 * 60 - 1", 2 * 60 - 1, 119)
    # rank example over F_8 = F_2[X]/(X^3 + X + 1), g = X: t = (X, X^2, X^2 + X), zeta = (0, 1)
    logs, x = {}, 1
    for e in range(7):
        logs[x] = e
        x <<= 1
        if x & 8:
            x ^= 0b1011
    cols = [[logs[t ^ z] for t in (0b010, 0b100, 0b110)] for z in (0, 1)]
    show("rank example over F_8: the two columns", "%s %s" % (tuple(cols[0]), tuple(cols[1])), "(1, 2, 4) (3, 6, 5)")
    show("rank example over F_8: second column = 3 * first modulo 7", all((3 * a - b) % 7 == 0 for a, b in zip(*cols)), True)


# (6) Table 5 -------------------------------------------------------------------
def lattice() -> None:
    print("(6) Table 5 (seed 3101): subsets completed and best mean cosine on all rows")
    table = (
        ("logcauchy-m600-s3101", 600, 100, 20, 50, 0.2172), ("logcauchy-m600-s3101", 600, 150, 20, 27, 0.2873),
        ("logcauchy-m2000-s3101", 2000, 180, 2, 50, 0.1213), ("logcauchy-m2000-s3101", 2000, 180, 20, 9, 0.1112),
        ("logcauchy-m2000-s3101", 2000, 220, 2, 50, 0.1521), ("logcauchy-m2000-s3101", 2000, 220, 20, 50, 0.1390),
        ("logcauchy-m2000-s3101", 2000, 260, 2, 8, 0.0816), ("logcauchy-m2000-s3101", 2000, 260, 20, 2, 0.0677),
        ("logcauchy-m6000-s3101", 6000, 470, 2, 50, 0.0942), ("logcauchy-m6000-s3101", 6000, 520, 2, 5, 0.0848),
    )
    for name, m, rows, block, subsets, score in table:
        rec = next(r for r in lattice_records(name) if r["rows"] == rows and r["block"] == block)
        label = "m = %d, %d rows, %s" % (m, rows, "LLL" if block == 2 else "BKZ-%d" % block)
        show(label + ": subsets", rec["subsets"], subsets)
        show(label + ": best mean cosine", rec["best_all_rows"], score, 4)
    for m, ell, paper in ((600, 20, 0.2235), (2000, 69, 0.2433), (6000, 209, 0.2518)):
        show("DQI (finite) at m = %d" % m, cosine_finite(m, ell), paper, 4)
    for seed, paper in ((3102, 0.257), (3103, 0.265)):
        show("m = 600, seed %d: best BKZ-20 mean cosine" % seed, max(r["best_all_rows"] for r in lattice_records("logcauchy-m600-s%d" % seed)), paper, 3)
    rec = next(r for r in lattice_records("logcauchy-m600-s3102") if r["rows"] == 150)
    show("m = 600, seed 3102: BKZ-20 on 150 rows", rec["best_all_rows"], 0.246, 3)
    for seed, paper in ((3102, 0.142), (3103, 0.126)):
        with open(os.path.join(RESULTS, "lattice", "run-m2000-s%d.log" % seed), encoding="utf-8") as fh:
            scores = [float(v) for v in re.findall(r"A rows=220 block=2: .*? all rows ([0-9.]+)", fh.read())]
        show("m = 2000, seed %d: LLL on 220 rows (from the log)" % seed, scores[0], paper, 3)
    for rec, paper in zip(lattice_records("logcauchy-m6000-s3101-bkz"), ((520, 0.0696, 2432), (700, 0.0192, 2600))):
        show("m = 6000, multiprecision BKZ-20, %d rows: mean cosine" % paper[0], rec["best_all_rows"], paper[1], 4)
        show("m = 6000, multiprecision BKZ-20, %d rows: seconds" % paper[0], rec["bkz_seconds"], paper[2], 0)
    slow = {(r["rows"], r["block"]): r["bkz_seconds"] for name in ("logcauchy-m2000-s3101", "logcauchy-m6000-s3101") for r in lattice_records(name)}
    show("m = 2000, BKZ-20 on 260 rows: seconds", slow[(260, 20)], 1596, 0)
    show("m = 6000, LLL on 470 rows: seconds", slow[(470, 2)], 1288, 0)
    show("m = 6000, LLL on 520 rows: seconds", slow[(520, 2)], 4060, 0)


# (7) rule (R4) -------------------------------------------------------------------------
def gaussian_binomial(b: int, k: int, p: int) -> int:
    num = den = 1
    for i in range(k):
        num *= p ** (b - i) - 1
        den *= p ** (i + 1) - 1
    return num // den


def expected_cosets(b: int, k: int, p: int = 2) -> float:
    """Expected number of affine k-spaces of F_p^b inside a uniformly random half-density set."""
    size, r = p ** b, p ** b // 2
    return gaussian_binomial(b, k, p) * p ** (b - k) * comb(size - p ** k, r - p ** k) / comb(size, r)


def subfield() -> None:
    print("(7) rule (R4): subfield examples")
    show("threshold 2c^2/(b^2 + c^2) at b = 4, c = 1, minus 2/17", 2 / 17 - 2 * 1 / (16 + 1), 0, 12)
    show("subfield baseline at alpha = 0.12: 1/2 + (1/2) min(1, 4 alpha)", 0.5 + 0.5 * min(1, 4 * 0.12), 0.74, 2)
    show("DQI at the Singleton reach alpha/2", q_dqi(0.06, 0.5), 0.7375, 4)
    show("expected affine planes in a half-density subset of F_32", expected_cosets(5, 2), 62.8, 1)
    show("expected 3-dimensional cosets in F_32", expected_cosets(5, 3), 0.76, 2)
    show("expected 4-dimensional cosets in F_256", expected_cosets(8, 4), 29.8, 1)
    show("expected 5-dimensional cosets in F_256, times 1e5", expected_cosets(8, 5) * 1e5, 2, 0)
    stuck = load("partially-stuck-finite-control-run.json")
    show("Reed-Solomon over F_256: rows of B", stuck["model"]["u_stuck_symbols"], 240)
    show("Reed-Solomon over F_256: columns of B", stuck["model"]["r_redundancy"], 28)
    ranks = {kind: sorted({v for r in stuck["results"] if r["identification"] == kind for v in r["affine_gf2"]["gf2_rank_of_single_bit_functionals"]})
             for kind in ("A_nibble", "C_nibble_grs_multipliers")}
    show("rank over F_2 with the same linear form on every row (3 seeds)", str(ranks["A_nibble"]), "[109]")
    show("rank over F_2 with random nonzero row multipliers (3 seeds)", str(ranks["C_nibble_grs_multipliers"]), "[224]")
    cosets, seen = 0, set()
    for e in range(28):  # exponents in the binary cyclotomic cosets modulo 255 that meet {0, ..., 27}
        if e not in seen:
            orbit = {(e * 2 ** i) % 255 for i in range(8)}
            seen |= orbit
            cosets += len(orbit)
    show("exponents in the cyclotomic cosets modulo 255 meeting {0, ..., 27}", cosets, 109)


# (8) solver budgets --------------------------------------------------------------------
def budgets() -> None:
    print("(8) Table 8: Prange completions per instance, seconds per search")
    for m, paper in ((400, 20000), (4000, 12500)):
        runs = opi_runs(m)
        show("OPI m = %d: Prange completions" % m, span([r["attacks"]["A_prange"]["restarts"] for r in runs], "%d"), str(paper))
    for m, seeds in ((14000, SEEDS), (20000, SEEDS)):
        runs = [alternant(m, s) for s in seeds]
        show("AMA m = %d: Prange completions" % m, span([r["attacks"]["A_prange"]["restarts"] for r in runs], "%d"), "1000")
        show("AMA m = %d: seconds per search" % m, span([r["budgets"]["line"] for r in runs], "%d"), "600")
        show("AMA m = %d: rows in the CP-SAT model" % m, span([r["attacks"]["E_cpsat"]["rows_modelled"] for r in runs], "%d"), "3000")
    for m, paper in ((2000, 125), (6000, 24)):
        runs = [multiplicative_run(m, s) for s in SEEDS]
        show("MPI m = %d: Prange completions" % m, span([r["attacks"]["A_prange"]["restarts"] for r in runs], "%d"), str(paper))
        show("MPI m = %d: seconds per search" % m, span([r["budgets"]["line"] for r in runs], "%d"), "600")
    show("MPI m = 6000, short runs: seconds per search", span([multiplicative_run(6000, s, "")["budgets"]["line"] for s in SEEDS], "%d"), "40")


# optional: re-score the stored assignments on instances regenerated from their seeds ----
def witnesses() -> None:
    print("(9) stored assignments: regenerate each instance from its seed, check Bx = c is solvable, re-score")
    from dqi_explorer.alternant_trapdoor import as_dense_arrays, build_alternant_trapdoor_instance
    from dqi_explorer.log_cauchy import build_log_cauchy_instance
    from dqi_explorer.screen import PrimeField, gauss_inverse

    def codeword(field, B, y, rng):
        n = B.shape[1]
        for _ in range(20):
            rows = np.sort(rng.choice(B.shape[0], n, replace=False))
            inv = gauss_inverse(field, B[rows])
            if inv is not None:
                x = field.matmul(inv, y[rows][:, None])
                return bool(np.array_equal(field.matmul(B, x)[:, 0], y))
        raise RuntimeError("no invertible row subset found")

    rng = np.random.default_rng(48)
    for path in sorted(glob.glob(os.path.join(RESULTS, "finite-control-alternant-*run-ext6*.json"))):
        rec = draft(os.path.basename(path))
        inst = build_alternant_trapdoor_instance(m=rec["m"], seed=rec["seed"], p=11, s=6, alpha=0.08)
        B, sets = as_dense_arrays(inst)
        if rec.get("matrix", "alternant key") != "alternant key":
            B = np.random.default_rng(rec["seed"] + 7_000_000).integers(0, 11, size=B.shape, dtype=np.int64)
        field = PrimeField(11)
        for name, attack in rec["attacks"].items():
            if "witness_y" not in attack:
                continue
            y = np.asarray(attack["witness_y"], dtype=np.int64)
            score = sum(int(v) in set(s.tolist()) for v, s in zip(y.tolist(), sets)) if codeword(field, B, y, rng) else -1
            show("%s %s" % (os.path.basename(path)[15:-5], name), score, attack["best"], ref="stored")
    for path in sorted(glob.glob(os.path.join(RESULTS, "finite-control-screen-logcauchy-run-*.json"))):
        rec = draft(os.path.basename(path))
        inst = build_log_cauchy_instance(m=rec["m"], seed=rec["seed"], h=rec["info"]["h"])
        B, c, p = np.asarray(inst.B_pub, dtype=np.int64), np.asarray(inst.centres, dtype=np.int64), inst.p
        field = PrimeField(p)
        for name, attack in rec["attacks"].items():
            if "witness_y" not in attack:
                continue
            y = np.asarray(attack["witness_y"], dtype=np.int64)
            score = float(np.cos(2 * np.pi * ((y - c) % p) / p).sum()) if codeword(field, B, y, rng) else -1.0
            show("%s %s" % (os.path.basename(path)[22:-5], name), score, round(attack["best"], 4), 4, ref="stored")


def main() -> None:
    print("OPI: optimal polynomial intersection; AMA: alternant max-agreement; MPI: multiplicative polynomial intersection\n")
    rules()
    experiments()
    candidate_rules()
    alternant_attacks()
    multiplicative()
    lattice()
    subfield()
    budgets()
    if "--witnesses" in sys.argv:
        witnesses()
    else:
        with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        print("written", os.path.relpath(OUT, ROOT))
    print("%d values, %d differ%s" % (len(record), len(differs), "" if not differs else ": " + "; ".join(differs)))
    sys.exit(1 if differs else 0)


if __name__ == "__main__":
    main()
