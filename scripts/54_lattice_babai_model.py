"""Script 54 - Calibrated Babai model of the row-subset lattice attack on MPI (numpy only).

Predicts the block size at which the row-subset CVP attack of script 43 (reduce the q-ary lattice of
m' rows with BKZ-beta, then Babai nearest plane to the centres) reaches a target mean cosine tau.

  - A BKZ-beta basis of the m'-dimensional M-ary lattice has Gram-Schmidt norms falling
    geometrically with ratio delta_beta^-2 (geometric series assumption), clamped to [1, M]
    (the q-ary Z-shape), with product M^(m'-n).
  - delta_beta: Chen-Nguyen experiments below beta = 50, the closed form from 50 on.
  - Babai leaves a residual uniform along each Gram-Schmidt direction (Lindner-Peikert). Directions
    at full length M are the axis vectors M e_i, so their rows score 0 on average; the other rows
    share the residual of the sloped part, taken as Gaussian per row, giving exp(-2 pi^2 s^2 / M^2)
    each. Rows outside the subset score 0 on average.
  - Maximising over m' gives the model value at beta. Over all our runs the measured best-of-run is
    at most 1.4 times the model value, so every prediction is multiplied by 1.4 (favours the attacker).

This is an order-of-magnitude estimate of one attack, not a bound. Usage:
  python scripts/54_lattice_babai_model.py --m 2000 --n 140 --p 8191 --tau 0.2433
  python scripts/54_lattice_babai_model.py --m 20000 --n 1400 --p 131071 --tau 0.2562
Writes results/lattice/logcauchy-params-m<m>-n<n>-p<p>.babai.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

SMALL_DELTA = {2: 1.0219, 10: 1.0144, 20: 1.0121, 30: 1.0110, 40: 1.0102}
CALIBRATION = 1.4


def gsa_delta(beta: int) -> float:
    """Root-Hermite factor BKZ-beta reaches (the standard asymptotic estimate)."""

    if beta < 20:
        return 1.0219
    return (beta / (2 * math.pi * math.e) * (math.pi * beta) ** (1.0 / beta)) ** (1.0 / (2 * (beta - 1)))


def delta_model(beta: int) -> float:
    if beta >= 50:
        return gsa_delta(beta)
    keys = sorted(SMALL_DELTA) + [50]
    return float(np.interp(beta, keys, [SMALL_DELTA[k] for k in sorted(SMALL_DELTA)] + [gsa_delta(50)]))


def qary_profile(d: int, logvol: float, p: int, delta: float) -> np.ndarray:
    """Log Gram-Schmidt norms of a BKZ-reduced q-ary basis under the GSA, clamped to [1, p]."""

    ld, lp = math.log(delta), math.log(p)
    slope = (d - 1 - 2 * np.arange(d)) * ld
    lo, hi = -d * ld - 1, lp + d * ld + 1
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.clip(mid + slope, 0.0, lp).sum() < logvol:
            lo = mid
        else:
            hi = mid
    return np.clip((lo + hi) / 2 + slope, 0.0, lp)


def babai_mean_cosine(m: int, n: int, p: int, rows: int, beta: int) -> float:
    """Expected mean cosine over all m rows of the attack on a rows-subset with a BKZ-beta basis."""

    prof = qary_profile(rows, (rows - n) * math.log(p), p, delta_model(beta))
    top = prof >= math.log(p) - 1e-9
    free = rows - int(top.sum())
    if free <= 0:
        return 0.0
    s2 = float(np.exp(2 * prof[~top]).sum()) / 12.0 / free
    return free / m * math.exp(-2 * math.pi ** 2 * s2 / p ** 2)


def babai_curve(m: int, n: int, p: int, tau: float) -> dict:
    """Block size at which the attack is predicted to reach tau, with the calibration applied."""

    grid_rows = np.unique(np.linspace(n + 1, m, 120).astype(int))
    curve, needed = [], None
    for beta in list(range(20, 100, 10)) + list(range(100, 2001, 50)):
        if beta > m:
            break
        score, rows = max((babai_mean_cosine(m, n, p, int(r), beta), int(r)) for r in grid_rows)
        point = {"beta": beta, "rows": rows, "model": score, "calibrated": CALIBRATION * score,
                 "core_svp_log2": 0.292 * beta}
        curve.append(point)
        if needed is None and CALIBRATION * score >= tau:
            needed = point
            break
    return {"attack": "E_babai_model", "tau": tau, "calibration": CALIBRATION, "curve": curve, "needed": needed,
            "note": "GSA Babai model of the row-subset attack (q-ary Z-shape, rows on q-vectors score 0); "
                    "calibrated by the largest measured best-of-run / model ratio; an estimate, not a bound"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=int, required=True, help="the modulus M = 2^h - 1")
    parser.add_argument("--tau", type=float, required=True, help="target mean cosine, e.g. DQI's exact value")
    args = parser.parse_args()
    res = babai_curve(args.m, args.n, args.p, args.tau)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lattice")
    os.makedirs(root, exist_ok=True)
    name = "logcauchy-params-m%d-n%d-p%d" % (args.m, args.n, args.p)
    out = os.path.join(root, name + ".babai.json")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"instance": name, "babai": res}, handle, indent=2)
        handle.write("\n")
    need = res["needed"]
    print("m=%d n=%d M=%d tau %.4f: %s" % (args.m, args.n, args.p, args.tau, "not reached by beta 2000" if need is None else
          "beta %d on %d rows (model %.4f, calibrated %.4f), core-SVP 2^%.0f"
          % (need["beta"], need["rows"], need["model"], need["calibrated"], need["core_svp_log2"])))
    print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
