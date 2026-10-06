"""Script 52 - Tables and figures for the scaling runs of scripts 50, 51 and 53.

Reads ``results/scaling/budget-*.json`` (script 50), ``lattice-*.json`` (script 51) and
``keyholder-*.json`` (script 53) and writes, under ``results/scaling/figures/``:

  summary.md            the tables of the paper's appendix on scaling;
  collapse.pdf          all families on one scale: scores in standard deviations above one Prange
                        completion, against the level reached by independent sampling;
  budget-<family>.pdf   classical score against the number of moves, one panel per length;
  lattice.pdf           the row-subset lattice attack: measured means against the block size, the
                        model, and the block size at which the model reaches DQI's value.

The common scale.  For a Prange completion the m - n rows outside the information set are
independent, each with variance sigma^2 (rho (1 - rho) for accepted sets of density rho, 1/2 for
the cosine), so the quality of a completion has standard deviation s = sigma sqrt((1 - alpha)/m)
around the Prange value.  Every quality is measured in these units, z = (Q - Q_Pr)/s.  The best
of N independent completions reaches the level zhat(N) defined by N P[Z >= zhat] = 1 for a
standard normal Z (to leading order sqrt(2 ln N)).  A search that makes N_mv moves scores
N = q N_mv codewords.

The lattice model.  After reduction the Gram-Schmidt norms of the M-ary lattice of m' rows follow
the usual shape: vectors of norm M, then norms falling geometrically, log2 |b_i*| decreasing by
2 log2(delta_0) per index, then vectors of norm 1, with total volume M^(m'-n).  Babai's
nearest-plane algorithm leaves a residual that is uniform in the Gram-Schmidt box, so the rows
off the plateau at M score exp(-2 pi^2 sigma^2 / M^2) each, with sigma^2 the mean of
|b_i*|^2 / 12 over those rows, and the plateau rows score zero on average.  The script checks
this against the stored profiles and scores, and then uses the standard estimate of delta_0 for
BKZ with block size beta to extrapolate.

The extrapolations describe the measured trends and the tested attacks; they are not bounds.

Run from the repository root (needs matplotlib):
  python scripts/52_scaling_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dqi_explorer.spectral import canonical_finite_quality  # noqa: E402

SCALING = os.path.join(ROOT, "results", "scaling")
FIGURES = os.path.join(SCALING, "figures")
FAMILIES = ("opi", "alternant", "logcauchy")
NAMES = {"opi": "optimal polynomial intersection (OPI)", "alternant": "alternant max-agreement (AMA), s = 6",
         "logcauchy": "multiplicative polynomial intersection (MPI)"}
SHORT = {"opi": "OPI", "alternant": "AMA", "logcauchy": "MPI"}
ABBREVIATIONS = "OPI: optimal polynomial intersection; AMA: alternant max-agreement; MPI: multiplicative polynomial intersection"
YLABEL = {"opi": "fraction of rows satisfied", "alternant": "fraction of rows satisfied", "logcauchy": "mean cosine"}
Q, ALPHA = 8191, 0.07  # the multiplicative family: modulus M and rate
MIN_SUBSETS = 6  # a (subset size, block size) cell of the lattice runs is reported from this many subsets on


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------
def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(SCALING, pattern))):
        with open(path, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def merged_budget() -> list[dict]:
    """The budget files, one record per instance: several files of one (family, m, seed) are merged."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in load("budget-*.json"):
        groups[(d["family"], d["m"], d["seed"])].append(d)
    out = []
    for _, files in sorted(groups.items()):
        d = dict(files[0])
        runs = {(r["kind"], r["moves"], r["chain"]): r for f in files for r in f["runs"]}
        d["runs"] = [runs[k] for k in sorted(runs)]
        out.append(d)
    return out


def merged_lattice() -> list[dict]:
    """The lattice files, one record per (m, seed)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in load("lattice-*.json"):
        groups[(d["m"], d["seed"])].append(d)
    out = []
    for _, files in sorted(groups.items()):
        d = dict(files[0])
        runs = {(r["rows"], r["subset"]): dict(r, block_seconds=f["block_seconds"]) for f in files for r in f["runs"]}
        d["runs"] = [runs[k] for k in sorted(runs)]
        out.append(d)
    return out


# ---------------------------------------------------------------------------------------
# the common scale
# ---------------------------------------------------------------------------------------
def variance(d: dict) -> float:
    """Variance of one row's objective: 1/2 for the cosine, rho (1 - rho) for accepted sets."""
    if d["family"] == "logcauchy":
        return 0.5
    rho = 5 / 11 if d["family"] == "alternant" else ((d["q"] - 1) // 2) / d["q"]
    return rho * (1 - rho)


def tail(z: float) -> float:
    """P[Z >= z] for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def sampling_level(count: float) -> float:
    """zhat(N): the level that the best of N independent completions reaches, N tail(zhat) = 1."""
    lo, hi = 0.0, 38.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if count * tail(mid) > 1 else (lo, mid)
    return hi


def log2_count(z: float) -> float:
    """log2 of the number N of independent completions whose best reaches z."""
    return -math.log2(tail(z))


def log2_cost_per_codeword(family: str, m: int, n: int, q: int) -> float:
    """log2 of the field operations per codeword scored.

    A move scores q codewords and updates the basis (m n operations).  For accepted sets of size r
    the scoring takes m r operations; for the cosine it takes m operations and one FFT of length q.
    """
    if family == "logcauchy":
        move = m * n + q * math.log2(q) + m
    else:
        r = 5 if family == "alternant" else (q - 1) // 2
        move = m * (r + n)
    return math.log2(move / q)


def instances() -> dict:
    """Per (family, m): DQI's value on the common scale and the classical searches.

    A run length is reported once every chain of every seed has finished.  Annealing: the mean
    over chains and seeds of the best quality visited, at the longest run length.  Line search:
    the best chain of each seed over all of its line-search chains, with the moves of those chains
    added up.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in merged_budget():
        groups[(d["family"], d["m"])].append(d)
    out = {}
    for (family, m), runs in sorted(groups.items()):
        d0 = runs[0]
        n, q, chains = d0["n"], d0["q"], d0["chains"]
        s = math.sqrt(variance(d0) * (1 - n / m) / m)
        inst = {"family": family, "m": m, "n": n, "q": q, "seeds": [d["seed"] for d in runs], "chains": chains,
                "dqi": d0["dqi_finite"], "prange": d0["prange"], "s": s, "z_dqi": (d0["dqi_finite"] - d0["prange"]) / s}
        inst["anneal"] = []
        for N in sorted({r["moves"] for d in runs for r in d["runs"] if r["kind"] == "anneal"}):
            per_seed = [[r["best"] / m for r in d["runs"] if r["kind"] == "anneal" and r["moves"] == N] for d in runs]
            if any(len(v) < chains for v in per_seed):
                continue
            allv = [v for seed in per_seed for v in seed]
            mean = float(np.mean([np.mean(v) for v in per_seed]))
            inst["anneal"].append({"N_mv": N, "mean": mean, "se": float(np.std(allv) / math.sqrt(len(allv))), "best": max(allv),
                                   "z": (mean - d0["prange"]) / s, "level": sampling_level(N * q)})
        lengths = [N for N in sorted({r["moves"] for d in runs for r in d["runs"] if r["kind"] == "line"})
                   if all(sum(r["kind"] == "line" and r["moves"] == N for r in d["runs"]) >= chains for d in runs)]
        if lengths:
            best = [max(r["curve"][-1][1] / m for r in d["runs"] if r["kind"] == "line" and r["moves"] in lengths) for d in runs]
            total = chains * sum(lengths)
            inst["line"] = {"lengths": lengths, "chains": chains * len(lengths), "total_moves": total, "best_per_seed": best,
                            "z_per_seed": [(b - d0["prange"]) / s for b in best], "z": (float(np.mean(best)) - d0["prange"]) / s,
                            "level": sampling_level(total * q)}
        if not inst["anneal"] and "line" not in inst:
            continue
        inst["log2_count_dqi"] = log2_count(inst["z_dqi"])
        inst["log2_ops_dqi"] = inst["log2_count_dqi"] + log2_cost_per_codeword(family, m, n, q)
        reached = [a["mean"] >= inst["dqi"] for a in inst["anneal"][-1:]]
        reached += [float(np.mean(inst["line"]["best_per_seed"])) >= inst["dqi"]] if "line" in inst else []
        inst["classical_reaches_dqi"] = any(reached)
        out[(family, m)] = inst
    return out


def extra_lengths() -> list[dict]:
    """Run lengths beyond the tabulated ones that are complete on only some of the seeds of an instance."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in merged_budget():
        groups[(d["family"], d["m"])].append(d)
    out = []
    for (family, m), runs in sorted(groups.items()):
        chains = runs[0]["chains"]
        for kind in ("anneal", "line"):
            value = (lambda r: r["best"] / m) if kind == "anneal" else (lambda r: r["curve"][-1][1] / m)
            lengths = sorted({r["moves"] for d in runs for r in d["runs"] if r["kind"] == kind})
            full = [N for N in lengths if all(sum(r["kind"] == kind and r["moves"] == N for r in d["runs"]) >= chains for d in runs)]
            for N in lengths:
                seeds = [d["seed"] for d in runs if sum(r["kind"] == kind and r["moves"] == N for r in d["runs"]) >= chains]
                if seeds and len(seeds) < len(runs) and N > max(full, default=0):
                    vals = [value(r) for d in runs if d["seed"] in seeds for r in d["runs"] if r["kind"] == kind and r["moves"] == N]
                    out.append({"family": family, "m": m, "kind": kind, "N_mv": N, "seeds": seeds, "mean": float(np.mean(vals)), "best": max(vals)})
    return out


def line_along_run(min_moves: int = 1000) -> dict:
    """Per (family, m): the line search along its run, from min_moves moves on, relative to independent sampling.

    "mean": range of (mean over chains of the best quality so far, in z) - zhat(q moves);
    "best": range of (best chain so far, in z) - zhat(chains q moves), with the seed and the
    number of moves at which the largest value occurs.  Each set of chains of one run length is
    followed separately.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in merged_budget():
        groups[(d["family"], d["m"])].append(d)
    out = {}
    for (family, m), runs in sorted(groups.items()):
        d0 = runs[0]
        n, q, chains = d0["n"], d0["q"], d0["chains"]
        s = math.sqrt(variance(d0) * (1 - n / m) / m)
        mean_dev, best_dev = [], []
        for d in runs:
            lines = [r for r in d["runs"] if r["kind"] == "line"]
            for length in sorted({r["moves"] for r in lines}):
                group = [r for r in lines if r["moves"] == length]
                if len(group) < chains:
                    continue
                for i, (moves, _) in enumerate(group[0]["curve"]):
                    if moves < min_moves:
                        continue
                    scores = [r["curve"][i][1] / m for r in group]
                    mean_dev.append((float(np.mean(scores)) - d0["prange"]) / s - sampling_level(q * moves))
                    best_dev.append(((max(scores) - d0["prange"]) / s - sampling_level(chains * q * moves), d["seed"], moves))
        if mean_dev:
            top = max(best_dev)
            out[(family, m)] = {"mean": [min(mean_dev), max(mean_dev)], "best": [min(best_dev)[0], top[0]], "best_seed": top[1], "best_moves": top[2]}
    return out


def alternant_lengths(lengths=(14000, 20000, 30000, 40000, 50000), eps: float = 0.01) -> dict:
    """Alternant keys (p = 11, s = 6, alpha = 0.08, accepted sets of size 5) at longer lengths."""
    p, s_ext, alpha, rho = 11, 6, 0.08, 5 / 11
    out = {}
    for m in lengths:
        r = round(alpha * m / s_ext)
        n, ell = s_ext * r, (r - 1) // 2
        Qm = canonical_finite_quality(m, p, 5, ell).value
        qpr = rho + (1 - rho) * n / m
        sd = math.sqrt(rho * (1 - rho) * (1 - n / m) / m)
        z, z_eps = (Qm - qpr) / sd, (Qm - eps - qpr) / sd
        cost = log2_cost_per_codeword("alternant", m, n, p)
        out[m] = {"r": r, "n": n, "ell": ell, "Q": Qm, "z": z, "log2_count": log2_count(z), "z_eps": z_eps,
                  "log2_count_eps": log2_count(z_eps), "log2_ops_eps": log2_count(z_eps) + cost}
    return out


def keyholder() -> list[dict]:
    """Script 53: best quality of each search, per seed."""
    out = []
    for d in load("keyholder-*.json"):
        best = defaultdict(list)
        for r in d["runs"]:
            best[r["kind"]].append(r["curve"][-1][1] / d["m"])
        out.append({"seed": d["seed"], "m": d["m"], "moves": d["moves"], "start": d["warm_start"] / d["m"], "dqi": d["dqi_finite"],
                    **{kind: max(v) for kind, v in best.items()}})
    return out


# ---------------------------------------------------------------------------------------
# the lattice attack: measurements and model
# ---------------------------------------------------------------------------------------
def cosine_finite(m: int, ell: int) -> float:
    A = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        A[k - 1, k] = A[k, k - 1] = np.sqrt(k * (m - k + 1))
    return float(np.linalg.eigvalsh(A)[-1]) / (np.sqrt(2) * m)


def delta_bkz(beta: float) -> float:
    """Root-Hermite factor of BKZ with block size beta (Chen's estimate; used for beta >= 50)."""
    return (beta / (2 * np.pi * np.e) * (np.pi * beta) ** (1 / beta)) ** (1 / (2 * (beta - 1)))


def score_from_profile(log2_norms, m: int) -> float:
    """Expected all-row score of Babai's point from a Gram-Schmidt profile."""
    r = 2.0 ** np.asarray(log2_norms, dtype=float)
    rest = r[r < 0.98 * Q]
    if len(rest) == 0:
        return 0.0
    sigma2 = float((rest ** 2).sum()) / (12 * len(rest))
    return len(rest) / m * float(np.exp(-2 * np.pi ** 2 * sigma2 / Q ** 2))


def model_profile(rows: int, n: int, s: float) -> np.ndarray:
    """log2 Gram-Schmidt norms: log2 M, then slope -s per index, then 0, with total (rows - n) log2 M."""
    lq, i = np.log2(Q), np.arange(rows)
    lo, hi = 0.0, lq + s * rows
    for _ in range(80):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.clip(c - s * i, 0, lq).sum() < (rows - n) * lq else (lo, c)
    return np.clip(hi - s * i, 0, lq)


def model_best(m: int, n: int, s: float) -> tuple[float, int, int]:
    """Best all-row score over the subset size, that size, and the number of vectors on the slope there."""
    lq = np.log2(Q)
    best = (0.0, 0, 0)
    for rows in range(n + 5, m + 1, max(1, m // 400)):
        prof = model_profile(rows, n, s)
        score = score_from_profile(prof, m)
        if score > best[0]:
            best = (score, rows, int(((prof > 1e-9) & (prof < lq - 1e-9)).sum()))
    return best


def slope_of(beta: float) -> float:
    return 2 * np.log2(delta_bkz(beta))


def beta_to_reach(m: int, n: int, target: float) -> dict:
    """Smallest block size (>= 50) at which the model reaches the target, with the slope length there.

    The estimate delta_bkz describes a geometric profile only while the block is shorter than the
    slope; "valid" records whether that holds at the returned block size.
    """
    lo, hi = 50.0, 20000.0
    if model_best(m, n, slope_of(lo))[0] >= target:
        return {"beta": lo, "at_most": True, "valid": True}
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if model_best(m, n, slope_of(mid))[0] >= target else (mid, hi)
    _, rows, slope = model_best(m, n, slope_of(hi))
    return {"beta": hi, "at_most": False, "rows": rows, "slope": slope, "valid": hi <= slope}


def largest_valid_block(m: int, n: int) -> dict:
    """The largest block size that is no longer than the slope of the model profile, and the model's score there."""
    lo, hi = 50.0, 20000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if model_best(m, n, slope_of(mid))[2] >= mid else (lo, mid)
    return {"beta": lo, "score": model_best(m, n, slope_of(lo))[0]}


def lattice_cells() -> dict:
    """Per m: for each (subset size, block size) the mean score over subsets (seeds pooled), with the model's value."""
    out = {}
    data = merged_lattice()
    for m in sorted({d["m"] for d in data}):
        runs = [d for d in data if d["m"] == m]
        steps = defaultdict(list)
        for d in runs:
            for r in d["runs"]:
                for st in r["steps"]:
                    if st["status"] == "ok":
                        steps[(r["rows"], st["block"])].append(dict(st, limit=r["block_seconds"], seed=d["seed"]))
        cells = {}
        for (rows, block), sts in sorted(steps.items()):
            if len(sts) < MIN_SUBSETS:
                continue
            vals = np.array([st["score_all_rows"] for st in sts])
            slopes, lengths = [], []
            for st in sts:
                prof = np.asarray(st["profile_log2"])
                decaying = np.nonzero((prof > 0.05) & (prof < np.log2(0.98 * Q)))[0]
                lengths.append(len(decaying))
                idx = decaying[3:-3]
                if len(idx) > 20:
                    slopes.append(-np.polyfit(idx, prof[idx], 1)[0])
            cells[(rows, block)] = {
                "mean": float(vals.mean()), "se": float(vals.std() / math.sqrt(len(vals))), "best": float(vals.max()), "subsets": len(vals),
                "model": float(np.mean([score_from_profile(st["profile_log2"], m) for st in sts])),
                "time_limited": sum(st["seconds"] >= st["limit"] for st in sts) > len(sts) / 2,
                "delta": float(2 ** (np.mean(slopes) / 2)) if slopes else None,
                "seeds": sorted({st["seed"] for st in sts}), "decaying": float(np.mean(lengths)),
                "above_M": float(np.mean([(np.asarray(st["profile_log2"]) > np.log2(Q) + 0.01).sum() for st in sts]))}
        out[m] = {"n": runs[0]["n"], "dqi": runs[0]["dqi_finite"], "seeds": sorted({d["seed"] for d in runs}), "cells": cells,
                  "tours": runs[0]["tours"], "block_seconds": runs[0]["block_seconds"]}
    return out


def lattice_best(cells: dict) -> dict:
    """For each block size, the subset size with the best mean."""
    out = {}
    for block in sorted({b for _, b in cells}):
        rows = max((c["mean"], r) for (r, b), c in cells.items() if b == block)[1]
        out[block] = dict(cells[(rows, block)], rows=rows)
    return out


def model_accuracy(lat: dict) -> dict:
    """Root-mean-square difference between measured means and the model evaluated on the stored profiles, per length."""
    out = {}
    for m, it in lat.items():
        err = np.array([c["mean"] - c["model"] for c in it["cells"].values()])
        out[m] = {"count": len(err), "rms": float(np.sqrt((err ** 2).mean())), "se": float(np.mean([c["se"] for c in it["cells"].values()])),
                  "largest_overestimate": float(-err.min()), "largest_underestimate": float(err.max())}
    return out


def hermite_factors(lat: dict) -> dict:
    """Measured root-Hermite factor per block size, averaged over the configurations with a fitted slope."""
    by = defaultdict(list)
    for it in lat.values():
        for (_, block), c in it["cells"].items():
            if c["delta"] and not c["time_limited"]:
                by[block].append(c["delta"])
    return {b: float(np.mean(v)) for b, v in sorted(by.items())}


def full_model_check(lat: dict, blocks=(50, 60)) -> dict:
    """The model with the estimated delta_0 (no measured profile) against the measured best means, at the largest block sizes."""
    out = {}
    for m, it in lat.items():
        best = lattice_best(it["cells"])
        for block in blocks:
            if block in best:
                out[(m, block)] = {"model": model_best(m, it["n"], slope_of(block))[0], "measured": best[block]["mean"], "se": best[block]["se"],
                                   "time_limited": best[block]["time_limited"]}
    return out


def lattice_need(lengths=(600, 800, 1000, 1500, 2000, 3000, 4000, 6000)) -> dict:
    """Per length: the block size at which the model reaches DQI's value, or where the estimate stops applying."""
    out = {}
    for m in lengths:
        n = int(round(ALPHA * m))
        dqi = cosine_finite(m, (n - 1) // 2)
        b = beta_to_reach(m, n, dqi)
        out[m] = dict(b, n=n, dqi=dqi)
        if not b["valid"]:
            out[m]["limit"] = largest_valid_block(m, n)
    return out


# ---------------------------------------------------------------------------------------
# summary tables
# ---------------------------------------------------------------------------------------
def f4(x: float, digits: int = 4) -> str:
    """Fixed-point with ties rounded up (scores are integers divided by m, so exact ties occur)."""
    return f"{x + 1e-9:.{digits}f}"


def fmt_range(values, digits: int = 4) -> str:
    lo, hi = f4(min(values), digits), f4(max(values), digits)
    return lo if lo == hi else f"{lo}-{hi}"


def summary(inst: dict, lat: dict) -> list[str]:
    L = ["# Scaling runs: summary", "",
         "Scores are fractions of rows satisfied (OPI, AMA) or mean cosines (MPI).",
         "z = (score - Prange) / (sigma sqrt((1 - alpha)/m)); N = q x moves is the number of codewords scored;",
         "zhat(N) is the level reached by the best of N independent completions, N P[Z >= zhat] = 1.", ""]
    L += ["## Budget scaling of the generic searches", "",
          "| family | m | seeds | DQI (finite) | z of DQI | annealing: moves per run | mean (best chain) | z | zhat(N) | line search: moves in total per seed (chains) | best chain per seed | z | zhat(N) | reaches DQI | log2 N for DQI | log2 field operations |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for family in FAMILIES:
        for (fam, m), it in inst.items():
            if fam != family:
                continue
            a = it["anneal"][-1] if it["anneal"] else None
            ln = it.get("line")
            L.append("| %s | %d | %d | %.4f | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.0f | %.0f |" % (
                NAMES[fam], m, len(it["seeds"]), it["dqi"], it["z_dqi"],
                "%.0e" % a["N_mv"] if a else "-", "%s (%s)" % (f4(a["mean"]), f4(a["best"])) if a else "-",
                "%.2f" % a["z"] if a else "-", "%.2f" % a["level"] if a else "-",
                "%.2g (%d)" % (ln["total_moves"], ln["chains"]) if ln else "-", fmt_range(ln["best_per_seed"]) if ln else "-",
                "%.2f" % ln["z"] if ln else "-", "%.2f" % ln["level"] if ln else "-",
                "yes" if it["classical_reaches_dqi"] else "no", it["log2_count_dqi"], it["log2_ops_dqi"]))
    L += ["", "## Annealing on the common scale: z of the mean minus zhat(N), for each run length", ""]
    for (fam, m), it in inst.items():
        L.append("- %s, m = %d: " % (NAMES[fam], m) + ", ".join("%.0e: %+.2f" % (a["N_mv"], a["z"] - a["level"]) for a in it["anneal"]))
    extra = extra_lengths()
    if extra:
        L += ["", "## Longer runs on some of the seeds only (not in the table above)", "",
              "| family | m | search | moves per chain | seeds | mean over chains | best chain |", "|---|---|---|---|---|---|---|"]
        for e in extra:
            L.append("| %s | %d | %s | %.0e | %s | %s | %s |" % (NAMES[e["family"]], e["m"], "annealing" if e["kind"] == "anneal" else "line search",
                                                           e["N_mv"], ", ".join(str(s) for s in e["seeds"]), f4(e["mean"]), f4(e["best"])))
    L += ["", "## Line search along the run (from 1000 moves on): z minus zhat(N)", "",
          "| family | m | mean over chains: lowest | highest | best chain: lowest | highest | at (seed, moves) |", "|---|---|---|---|---|---|---|"]
    for (fam, m), a in line_along_run().items():
        L.append("| %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %d, %d |" % (NAMES[fam], m, *a["mean"], *a["best"], a["best_seed"], a["best_moves"]))
    L += ["", "## AMA keys at longer lengths (p = 11, s = 6, alpha = 0.08)", "",
          "| m | r | l | DQI (finite) | z | log2 N | z of DQI - 0.01 | log2 N | log2 field operations |", "|---|---|---|---|---|---|---|---|---|"]
    for m, a in alternant_lengths().items():
        L.append("| %d | %d | %d | %.4f | %.2f | %.0f | %.2f | %.0f | %.0f |" % (
            m, a["r"], a["ell"], a["Q"], a["z"], a["log2_count"], a["z_eps"], a["log2_count_eps"], a["log2_ops_eps"]))
    keys = keyholder()
    if keys:
        L += ["", "## AMA, m = 14000: searches with the key and a search without it, same start, same number of moves", "",
              "| seed | moves | start | with the key, line search | with the key, annealing | with the key, random start | without the key, line search | DQI (finite) |",
              "|---|---|---|---|---|---|---|---|"]
        for k in keys:
            L.append("| %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
                k["seed"], k["moves"], k["start"], k["key_line_warm"], k["key_heat_warm"], k["key_line_cold"], k["keyless_warm"], k["dqi"]))
    if lat:
        L += ["", "## Row-subset lattice attack on MPI: mean over subsets at the best subset size", "",
              "(*: more than half of the subsets reached the time limit at this block size)", "",
              "| m | n | seeds | DQI (finite) | block | rows | subsets | mean | standard error | best subset | model on the stored profiles |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, it in lat.items():
            for block, c in lattice_best(it["cells"]).items():
                L.append("| %d | %d | %d | %.4f | %d%s | %d | %d | %.4f | %.4f | %.4f | %.4f |" % (
                    m, it["n"], len(it["seeds"]), it["dqi"], block, "*" if c["time_limited"] else "", c["rows"], c["subsets"], c["mean"],
                    c["se"], c["best"], c["model"]))
        L += ["", "## Lattice model against the measurements, all configurations", "",
              "| m | configurations | rms difference | mean standard error | largest overestimate by the model | largest underestimate |",
              "|---|---|---|---|---|---|"]
        for m, a in model_accuracy(lat).items():
            L.append("| %d | %d | %.4f | %.4f | %.4f | %.4f |" % (m, a["count"], a["rms"], a["se"], a["largest_overestimate"], a["largest_underestimate"]))
        L += ["", "Measured root-Hermite factors (slope of the profiles; at most %d tours per block size): " % next(iter(lat.values()))["tours"]
              + ", ".join("%s %.4f" % ("LLL" if b == 2 else "BKZ-%d" % b, d) for b, d in hermite_factors(lat).items()),
              "Estimate used for larger block sizes: " + ", ".join("BKZ-%d %.4f" % (b, delta_bkz(b)) for b in (50, 60)), "",
              "The model with the estimated factor against the measured means: "
              + "; ".join("m = %d, BKZ-%d: %.3f against %.3f" % (m, b, c["model"], c["measured"]) for (m, b), c in full_model_check(lat).items())]
    L += ["", "## Lattice model: block size at which the modeled attack reaches DQI's value (M = 8191, alpha = 0.07)", "",
          "| m | n | DQI (finite) | block size | vectors on the slope | 0.292 x block size |", "|---|---|---|---|---|---|"]
    for m, b in lattice_need().items():
        if b["at_most"]:
            L.append("| %d | %d | %.4f | at most 50 | - | - |" % (m, b["n"], b["dqi"]))
        elif b["valid"]:
            L.append("| %d | %d | %.4f | %.0f | %d | %.0f |" % (m, b["n"], b["dqi"], b["beta"], b["slope"], 0.292 * b["beta"]))
        else:
            L.append("| %d | %d | %.4f | not reached while the block is shorter than the slope (model: %.4f at block size %.0f) | - | above %.0f |" % (
                m, b["n"], b["dqi"], b["limit"]["score"], b["limit"]["beta"], 0.292 * b["limit"]["beta"]))
    return L


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
def collapse_figure(inst: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"opi": "tab:green", "alternant": "tab:blue", "logcauchy": "tab:orange"}
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    seen = defaultdict(int)
    for family in FAMILIES:
        for (fam, m), it in inst.items():
            if fam != family:
                continue
            marker = markers[seen[fam] % len(markers)]
            seen[fam] += 1
            ax.plot([a["level"] for a in it["anneal"]], [a["z"] for a in it["anneal"]], marker=marker, ms=4, lw=1, color=colors[fam],
                    label=f"{SHORT[fam]}, m = {m}")
            if "line" in it:
                ax.plot([it["line"]["level"]] * len(it["line"]["z_per_seed"]), it["line"]["z_per_seed"], marker=marker, ms=5, mfc="none", ls="",
                        color=colors[fam])
            if it["z_dqi"] < 12.3:
                ax.axhline(it["z_dqi"], color=colors[fam], ls=":", lw=0.8)
                ax.text(7.25, it["z_dqi"], f"DQI, m = {m}", color=colors[fam], fontsize=6.5, va="bottom", ha="right")
    ax.plot([2.0, 7.3], [2.0, 7.3], color="gray", lw=1)
    ax.set_xlim(2.0, 7.3)
    ax.set_ylim(1.5, 12.5)
    ax.set_xlabel("zhat(N): level reached by the best of N independent completions")
    ax.set_ylabel("z: standard deviations above one Prange completion")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.set_title("filled: annealing (mean over chains and seeds); open: line search (best chain of each seed)\n" + ABBREVIATIONS, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "collapse.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIGURES, "collapse.png"), dpi=150)
    plt.close(fig)


def budget_figures() -> None:
    by_family: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for d in merged_budget():
        by_family[d["family"]][d["m"]].append(d)
    for family, sizes in by_family.items():
        fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 3.8), squeeze=False)
        for ax, m in zip(axes[0], sorted(sizes)):
            runs = sizes[m]
            colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(runs)))
            for d, color in zip(runs, colors):
                by_n = defaultdict(list)
                for r in d["runs"]:
                    if r["kind"] == "anneal":
                        by_n[r["moves"]].append(r["best"] / m)
                xs = sorted(by_n)
                if xs:
                    ax.errorbar(xs, [np.mean(by_n[x]) for x in xs], yerr=[np.std(by_n[x]) for x in xs], color=color, marker="o", ms=3.5,
                                lw=1.2, capsize=2, label=f"annealing, seed {d['seed']}")
                lines = [r for r in d["runs"] if r["kind"] == "line"]
                if lines:
                    longest = max(r["moves"] for r in lines)
                    group = [r for r in lines if r["moves"] == longest]
                    ax.plot([c[0] for c in group[0]["curve"]], [np.mean([r["curve"][i][1] / m for r in group]) for i in range(len(group[0]["curve"]))],
                            color=color, ls="--", lw=1.0, label=f"line search, seed {d['seed']}")
            ax.axhline(runs[0]["dqi_finite"], color="crimson", lw=1.4)
            ax.axhline(runs[0]["prange"], color="gray", lw=1.0)
            ax.text(0.02, runs[0]["dqi_finite"], " DQI", color="crimson", va="bottom", transform=ax.get_yaxis_transform(), fontsize=8)
            ax.text(0.02, runs[0]["prange"], " one Prange completion", color="gray", va="bottom", transform=ax.get_yaxis_transform(), fontsize=8)
            ax.set_xscale("log")
            ax.set_xlim(left=50)
            ax.set_xlabel("moves per run")
            ax.set_ylabel(YLABEL[family])
            ax.set_title(f"m = {m}", fontsize=10)
            ax.legend(fontsize=6.5, loc="center right")
        fig.suptitle(NAMES[family] + ": annealing (mean and spread over chains) and line search (mean over chains)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, f"budget-{family}.pdf"), metadata={"CreationDate": None})
        fig.savefig(os.path.join(FIGURES, f"budget-{family}.png"), dpi=150)
        plt.close(fig)


def model_curve(m: int, n: int, betas) -> list[tuple[float, float]]:
    """The model's score against the block size, while the block is no longer than the slope."""
    pts = []
    for b in betas:
        score, _, slope = model_best(m, n, slope_of(b))
        if b <= slope:
            pts.append((float(b), score))
    return pts


def need_curve(lengths) -> list[tuple[int, float]]:
    pts = []
    for m in lengths:
        n = int(round(ALPHA * m))
        b = beta_to_reach(int(m), n, cosine_finite(int(m), (n - 1) // 2))
        if not b["at_most"] and b["valid"]:
            pts.append((int(m), b["beta"]))
    return pts


def lattice_figure(lat: dict) -> None:
    if not lat:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    betas = np.unique(np.round(np.geomspace(50, 1500, 30)))
    for m, color in zip(lat, plt.cm.viridis(np.linspace(0.1, 0.85, len(lat)))):
        it = lat[m]
        best = lattice_best(it["cells"])
        ax.errorbar(list(best), [c["mean"] for c in best.values()], yerr=[c["se"] for c in best.values()], fmt="o", color=color, ms=4, capsize=2,
                    label=f"m = {m}")
        curve = model_curve(m, it["n"], betas)
        ax.plot([b for b, _ in curve], [s for _, s in curve], "-", color=color, lw=1.1)
        ax.axhline(it["dqi"], color=color, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("BKZ block size (2: LLL)")
    ax.set_ylabel("mean cosine on all rows")
    ax.set_title("MPI (multiplicative polynomial intersection)\nmarks: measured; curves: model; dotted: DQI", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    pts = need_curve(np.unique(np.round(np.geomspace(700, 3200, 16) / 10) * 10).astype(int))
    ax.plot([m for m, _ in pts], [b for _, b in pts], "-", color="tab:red")
    ax.set_xlabel("length m (alpha = 0.07, M = 8191)")
    ax.set_ylabel("block size at which the model reaches DQI's value")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "lattice.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIGURES, "lattice.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    inst, lat = instances(), lattice_cells()
    lines = summary(inst, lat)
    with open(os.path.join(FIGURES, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    collapse_figure(inst)
    budget_figures()
    lattice_figure(lat)
    print("\nwrote", os.path.relpath(FIGURES, ROOT))


if __name__ == "__main__":
    main()
