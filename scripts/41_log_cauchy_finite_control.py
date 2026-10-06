"""Script 41 - Classical portfolio against finite DQI for multiplicative polynomial intersection.

Per (size, seed), through ``dqi_explorer/screen.py``:
  1. build ``B[i][j] = log_g(t_i + alpha_j)`` over ``F_M``, ``M = 2^13 - 1``, with random centres;
  2. verify the public restricted-error decoder (random +-1 patterns, plus the all-plus and
     all-minus worst cases of script 40) and compute DQI's exact finite expected mean cosine,
     the Prange baseline ``alpha`` and the Prange-restart threshold (objective range 2);
  3. attacks on the public matrix: A Prange completions solved to the centres, B pivoting line
     search (FFT over all M candidates), C simulated annealing (heat bath); ``--bkz`` adds BKZ
     on random row subsets (needs fpylll) and least-squares rounding, which the paper's
     portfolio does not use (its lattice runs are script 43);
     every assignment is re-scored on all rows;
  4. write ``results/finite-control-screen-logcauchy-run-[tag-]m{m}-s{seed}.json``.

Run from the repository root:
  python scripts/41_log_cauchy_finite_control.py --sizes 2000 --quick
  python scripts/41_log_cauchy_finite_control.py --sizes 6000 --seeds 3101
  python scripts/41_log_cauchy_finite_control.py --merge
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXAMPLES)
EXPLORER = os.path.join(ROOT, "src")
sys.path.insert(0, EXPLORER)

from dqi_explorer.log_cauchy import build_log_cauchy_instance, verify_decoder  # noqa: E402
from dqi_explorer.screen import Budgets, draft_path, gauss_inverse, merge, screen_one, verdict, write_draft  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


H = int(arg("--h", "13"))
ALPHA = float(arg("--alpha", "0.07"))
QUICK = "--quick" in sys.argv
SIZES = [int(v) for v in arg("--sizes", "2000,6000").split(",")]
SEEDS = [int(v) for v in arg("--seeds", "3101,3102,3103").split(",")]
if QUICK:
    SEEDS = SEEDS[:1]
BKZ_SECONDS = float(arg("--bkz-seconds", "20" if QUICK else "120"))
BKZ_ROWS = int(arg("--bkz-rows", "0"))  # 0: n + 60
BKZ_BLOCK = int(arg("--bkz-block", "20"))
USE_BKZ = "--bkz" in sys.argv  # off by default: the paper's portfolio is A, B, C; lattice attacks are script 43
T_LINE = float(arg("--line-seconds", "40"))
T_HEAT = float(arg("--heat-seconds", "40"))
TAG = arg("--tag", "")
RESULTS_DIR = os.path.join(ROOT, "results")
FAMILY = "logcauchy"


def attack_bkz(ctl, inst) -> tuple[dict, np.ndarray | None]:
    """BKZ on the q-ary lattice {v : v = B_R x mod M} of a random row subset R, Babai to the centres."""

    try:
        from fpylll import BKZ, GSO, LLL, IntegerMatrix
        from fpylll.fplll.bkz_param import BKZParam
    except Exception as exc:
        return {"skipped": f"fpylll unavailable: {type(exc).__name__}"}, None
    p, m, n = inst.p, inst.m, inst.n
    rows_wanted = BKZ_ROWS or min(m, n + 60)
    t_end = time.time() + BKZ_SECONDS
    best, best_y, subsets = None, None, 0
    field = ctl.f
    while time.time() < t_end and subsets < 20:
        R = np.sort(ctl.rng.choice(m, rows_wanted, replace=False))
        BR = inst.B[R]
        # basis rows: the n columns of B_R (as integer vectors) and M e_i
        basis = IntegerMatrix(n + rows_wanted, rows_wanted)
        for j in range(n):
            for i in range(rows_wanted):
                basis[j, i] = int(BR[i, j])
        for i in range(rows_wanted):
            basis[n + i, i] = p
        LLL.reduction(basis)
        # drop the zero rows LLL leaves at the top
        nonzero = [r for r in range(basis.nrows) if any(basis[r, c] != 0 for c in range(basis.ncols))]
        reduced = IntegerMatrix(len(nonzero), rows_wanted)
        for r_new, r_old in enumerate(nonzero):
            for c in range(rows_wanted):
                reduced[r_new, c] = basis[r_old, c]
        try:
            BKZ.reduction(reduced, BKZParam(block_size=min(BKZ_BLOCK, reduced.nrows), max_time=max(1.0, t_end - time.time())))
        except Exception:
            pass
        M = GSO.Mat(reduced)
        M.update_gso()
        target = [int(c) for c in inst.centres[R]]
        coeffs = M.babai(target)
        v = np.zeros(rows_wanted, dtype=np.int64)
        for r in range(reduced.nrows):
            if coeffs[r]:
                for c in range(rows_wanted):
                    v[c] += coeffs[r] * reduced[r, c]
        v %= p
        # recover x from an invertible n-subset of R
        for _ in range(10):
            T = np.sort(ctl.rng.choice(rows_wanted, n, replace=False))
            inv = gauss_inverse(field, BR[T])
            if inv is not None:
                break
        else:
            continue
        x = field.matmul(inv, v[T][:, None])[:, 0]
        y = field.matmul(inst.B, x[:, None])[:, 0]
        s = ctl.score(y)
        subsets += 1
        if best is None or s > best:
            best, best_y = s, y.copy()
    if best_y is None:
        return {"skipped": "no subset processed in the budget"}, None
    return {"best": best, "subsets": subsets, "rows_per_subset": rows_wanted, "block_size": BKZ_BLOCK, "seconds": BKZ_SECONDS, "note": "BKZ on a random row subset, Babai to the centres; not a full-length lattice attack"}, best_y


def attack_rounding(ctl, inst) -> tuple[dict, np.ndarray | None]:
    """Least-squares rounding of B x toward the centres for random integer lifts (no lattice reduction).

    The fallback when fpylll is missing: it measures what rounding alone gives (script 30's F_babai
    for the ISIS-infinity form).  A negative result here says nothing about reduced-lattice attacks.
    """

    p, m, n = inst.p, inst.m, inst.n
    B = inst.B.astype(np.float64)
    pinv = np.linalg.pinv(B)
    best, best_y = None, None
    for trial in range(32):
        lift = ctl.rng.integers(-1, 2, m) if trial else np.zeros(m)
        target = inst.centres.astype(np.float64) + p * lift
        x = np.rint(pinv @ target).astype(np.int64) % p
        y = ctl.f.matmul(inst.B, x[:, None])[:, 0]
        s = ctl.score(y)
        if best is None or s > best:
            best, best_y = s, y.copy()
    return {"best": best, "trials": 32, "lll_used": False, "note": "least-squares rounding toward the centres; no lattice reduction (fpylll unavailable)"}, best_y


def build(m: int, seed: int):
    return build_log_cauchy_instance(m=m, seed=seed, h=H, alpha=ALPHA)


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    budgets = Budgets.quick() if QUICK else Budgets(decoder_trials=30, line_seconds=T_LINE, heat_seconds=T_HEAT)
    budgets.law = False
    extra = (("F_bkz", attack_bkz), ("F_rounding", attack_rounding)) if USE_BKZ else ()
    tag = "quick" if QUICK else TAG
    records = []
    for m in SIZES:
        for seed in SEEDS:
            inst = build(m, seed)
            worst = verify_decoder(inst, 0, seed + 3, beyond_trials=4)  # the all-plus / all-minus cases only
            rec = screen_one(inst, m, seed, objective="cos", budgets=budgets, family=FAMILY, extra_attacks=extra)
            rec["decoder_worst_cases"] = worst
            print(f"   worst-case patterns: plus {worst['by_kind']['plus']} minus {worst['by_kind']['minus']} beyond-radius all-plus {worst['beyond_radius']}", flush=True)
            records.append(rec)
            path = draft_path(RESULTS_DIR, FAMILY, m, seed, tag)
            write_draft(path, [rec], family=FAMILY, objective="cos", extra={"generator": "dqi_explorer.log_cauchy:build_log_cauchy_instance", "h": H, "alpha": ALPHA, "line_seconds": T_LINE, "heat_seconds": T_HEAT, "bkz": {"seconds": BKZ_SECONDS, "block": BKZ_BLOCK, "rows": BKZ_ROWS}})
            print("   wrote", os.path.relpath(path, ROOT), flush=True)
    for key, v in verdict(records).items():
        print(f"   -> {key}: {v['reason']}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        records = merge(RESULTS_DIR, FAMILY)
        for rec in sorted(records, key=lambda r: (r["m"], r["seed"])):
            attacks = "  ".join(f"{k}:{v['fraction']:.4f}" for k, v in rec["attacks"].items() if "fraction" in v)
            print(f"m={rec['m']} seed={rec['seed']} alpha={rec['alpha']:.3f} ell={rec['ell']} Q_fin {rec['Q_fin']['value']:.4f} Q_Pr {rec['Q_Pr']:.4f} threshold {rec['threshold']:.0f} | {attacks} | best {rec['best_classical'] / rec['m']:.4f} ({'beaten' if rec['beats_dqi'] else 'below DQI'})")
        for key, v in verdict(records).items():
            print(f"   -> {key}: {v['reason']} ({len(v['seeds'])} runs)")
    else:
        run()
