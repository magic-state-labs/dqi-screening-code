"""Script 51 - Scaling of the row-subset lattice attack on multiplicative polynomial intersection.

Script 43 runs the attack inside a wall-clock budget.  This script runs it on a fixed number of
random row subsets and a ladder of block sizes, so that the score can be read as a function of the
block size and of the subset size:

  for each subset size d and each of ``--subsets`` random subsets R of d rows
    build the M-ary lattice {v : v = B_R x mod M} (systematic basis) and LLL-reduce it;
    for each block size b in ``--blocks`` (increasing; 2 means LLL alone)
      continue the reduction with BKZ-b (BKZ 2.0, at most ``--tours`` tours, ``--block-seconds`` s);
      Babai's nearest-plane algorithm towards the centres c_R gives a lattice vector, hence x;
      score x on ALL m rows (mean cosine);
      optionally improve x by line search (``--polish-moves`` moves, the moves of script 50).

For each (d, b) the script reports the mean over subsets (an unbiased estimate of what one
subset gives) and the best over subsets (which benefits from selection among the subsets).
The Gram-Schmidt profile after each block size is stored for comparison with a BKZ simulator.

Floating-point type: double up to dimension 160, then long double, double-double and quad-double
when fplll provides them (Linux builds from conda-forge do), otherwise multiprecision.

Run from the repository root (needs fpylll):
  python scripts/51_lattice_scaling.py --size 600 --seed 3101 --rows 100,150,200 --blocks 2,20,30,40 --subsets 8
Writes results/scaling/lattice-m<m>-s<seed>[-tag].json.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import json
import multiprocessing
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fpylll import BKZ, FPLLL, GSO, LLL, IntegerMatrix  # noqa: E402
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2  # noqa: E402
from fpylll.config import float_types  # noqa: E402

_spec = importlib.util.spec_from_file_location("budget_scaling", os.path.join(ROOT, "scripts", "50_budget_scaling.py"))
scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scaling)

_path = BKZ.DEFAULT_STRATEGY.decode() if isinstance(BKZ.DEFAULT_STRATEGY, bytes) else str(BKZ.DEFAULT_STRATEGY)
STRATEGIES = _path if os.path.exists(_path) else None  # pruning strategies; some pip wheels ship without them
INSTANCE: dict = {}
ARGS = None


def float_type(d: int) -> str:
    """Enough precision for size reduction in dimension d: 53, 64, 106, 212 bits, then multiprecision."""
    for limit, name in ((160, "d"), (250, "ld"), (450, "dd"), (800, "qd")):
        if d <= limit and name in float_types:
            return name
    FPLLL.set_precision(max(150, d // 3))
    return "mpfr"


def babai(A, M, target: list[int]):
    """Nearest-plane coefficients; fplll implements this for some float types only, so fall back to multiprecision."""
    try:
        return M.babai(target)
    except NotImplementedError:
        FPLLL.set_precision(max(150, A.nrows // 3))
        M2 = GSO.Mat(A, float_type="mpfr")
        M2.update_gso()
        return M2.babai(target)


def systematic_basis(BR: np.ndarray, p: int, rng) -> np.ndarray:
    """Basis of {v : v = B_R x mod p}: rows (e_j on T | column j of B_rest B_T^-1) and p e_i off T."""
    d, n = BR.shape
    field = scaling.PrimeField(p)
    while True:
        T = np.sort(rng.choice(d, n, replace=False))
        inv = scaling.inverse_mod(BR[T], p, field)
        if inv is not None:
            break
    rest = np.setdiff1d(np.arange(d), T)
    A = field.matmul(BR[rest], inv)
    basis = np.zeros((d, d), dtype=np.int64)
    basis[np.arange(n), T] = 1
    basis[np.arange(n)[:, None], rest[None, :]] = A.T
    basis[n + np.arange(d - n), rest] = p
    return basis


def solve_x(BR: np.ndarray, v: np.ndarray, p: int, rng) -> np.ndarray | None:
    n, field = BR.shape[1], scaling.PrimeField(p)
    for _ in range(20):
        T = np.sort(rng.choice(BR.shape[0], n, replace=False))
        inv = scaling.inverse_mod(BR[T], p, field)
        if inv is not None:
            return field.matmul(inv, (v[T] % p)[:, None])[:, 0]
    return None


def polish(ctl, y: np.ndarray, max_moves: int) -> float:
    """Line search (the moves of script 50) started from the codeword y, no restarts."""
    obj, p = ctl.obj, ctl.q
    while True:
        S = ctl.rng.choice(ctl.m, ctl.n, replace=False)
        inv = scaling.inverse_mod(ctl.B[S], p, ctl.f)
        if inv is not None:
            break
    L = scaling.compact(ctl.f.matmul(ctl.B, inv), p)
    cur, stale, moves = obj.score(y), 0, 0
    while stale < scaling.PATIENCE and moves < max_moves:
        k = int(ctl.rng.integers(ctl.n))
        u = L[:, k].astype(np.int64)
        counts = obj.line_scores(y, u, ctl.f)
        top = counts.max()
        moves += 1
        if obj.better(top, cur):
            t = int(ctl.rng.choice(np.nonzero(counts >= top - 1e-9)[0]))
            y = ctl.f.add(y, ctl.f.mul(t, u))
            cur, stale = float(counts[t]), 0
        else:
            stale += 1
        S, L = scaling.pivot(ctl, S, L, k, obj.satisfied(y))
    return cur


def run_subset(task: tuple) -> dict:
    d, k = task
    inst, args = INSTANCE, ARGS
    B, c, p = inst["B"], inst["centres"], inst["p"]
    m = B.shape[0]
    rng = np.random.default_rng([args.seed, d, k])
    R = np.sort(rng.choice(m, d, replace=False))
    A = IntegerMatrix.from_matrix(systematic_basis(B[R], p, rng).tolist())
    t0 = time.time()
    LLL.reduction(A)
    ft = float_type(d)
    out = {"rows": d, "subset": k, "float_type": ft, "steps": []}
    for block in args.block_list:
        t1 = time.time()
        M = GSO.Mat(A, float_type=ft)
        M.update_gso()
        status = "ok"
        if block > 2:
            params = {"block_size": min(block, d), "max_loops": args.tours, "max_time": args.block_seconds,
                      "flags": BKZ.MAX_LOOPS | BKZ.MAX_TIME | BKZ.AUTO_ABORT}
            if STRATEGIES is not None:
                params["strategies"] = STRATEGIES
            try:
                BKZ2(M)(BKZ.Param(**params))
            except Exception as exc:  # numerical failure: keep the basis reached so far
                status = "%s: %s" % (type(exc).__name__, exc)
            M = GSO.Mat(A, float_type=ft)
            M.update_gso()
        profile = [round(0.5 * float(np.log2(M.get_r(i, i))), 3) for i in range(d)]
        coeffs = np.array(babai(A, M, [int(v) for v in c[R]]), dtype=object)
        v = np.array(coeffs.dot(np.array([[A[i, j] for j in range(d)] for i in range(d)], dtype=object)) % p, dtype=np.int64)
        x = solve_x(B[R], v, p, rng)
        y = scaling.PrimeField(p).matmul(B, x[:, None])[:, 0]
        cos = np.cos(2 * np.pi * ((y - c) % p) / p)
        step = {"block": block, "status": status, "score_all_rows": float(cos.mean()), "score_subset": float(cos[R].mean()),
                "seconds": time.time() - t1, "profile_log2": profile}
        if args.polish_moves:
            ctl = scaling.control(inst, rng)
            step["score_after_line_search"] = polish(ctl, y, args.polish_moves) / m
        out["steps"].append(step)
    out["seconds"] = time.time() - t0
    return out


def summarize(done: list[dict]) -> list[dict]:
    rows = []
    for d in sorted({r["rows"] for r in done}):
        for block in ARGS.block_list:
            steps = [s for r in done if r["rows"] == d for s in r["steps"] if s["block"] == block]
            if not steps:
                continue
            vals = [s["score_all_rows"] for s in steps]
            row = {"rows": d, "block": block, "subsets": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                   "best": float(np.max(vals)), "seconds_mean": float(np.mean([s["seconds"] for s in steps])),
                   "failures": sum(s["status"] != "ok" for s in steps)}
            if "score_after_line_search" in steps[0]:
                pol = [s["score_after_line_search"] for s in steps]
                row.update({"line_search_mean": float(np.mean(pol)), "line_search_best": float(np.max(pol))})
            rows.append(row)
    return rows


def main() -> None:
    global INSTANCE, ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rows", required=True, help="subset sizes, comma separated")
    parser.add_argument("--blocks", default="2,20,30,40", help="block sizes, increasing; 2 means LLL alone")
    parser.add_argument("--subsets", type=int, default=8)
    parser.add_argument("--tours", type=int, default=8)
    parser.add_argument("--block-seconds", type=float, default=3600.0)
    parser.add_argument("--polish-moves", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--tag", default="")
    ARGS = parser.parse_args()
    ARGS.block_list = sorted(int(b) for b in ARGS.blocks.split(","))
    INSTANCE = scaling.build("logcauchy", ARGS.size, ARGS.seed)
    m, n = INSTANCE["B"].shape
    print(f"logcauchy m={m} n={n} seed={ARGS.seed}: DQI {INSTANCE['dqi']:.4f}; strategies {'yes' if STRATEGIES else 'no'}; float types {float_types}", flush=True)
    tasks = [(int(d), k) for d in ARGS.rows.split(",") for k in range(ARGS.subsets)]
    out_dir = os.path.join(ROOT, "results", "scaling")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"lattice-m{m}-s{ARGS.seed}{'-' + ARGS.tag if ARGS.tag else ''}.json")
    done: list[dict] = []

    def save() -> None:
        payload = {"family": "logcauchy", "m": m, "n": n, "q": INSTANCE["p"], "seed": ARGS.seed, "dqi_finite": INSTANCE["dqi"],
                   "prange": INSTANCE["prange"], "tours": ARGS.tours, "block_seconds": ARGS.block_seconds,
                   "polish_moves": ARGS.polish_moves, "strategies": STRATEGIES is not None, "summary": summarize(done), "runs": done}
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh)
            fh.write("\n")
        os.replace(path + ".tmp", path)

    with multiprocessing.get_context("fork").Pool(min(ARGS.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(run_subset, tasks, chunksize=1):
            done.append(res)
            print(f"   rows {res['rows']:4d} subset {res['subset']:2d} ({res['float_type']}, {res['seconds']:.0f}s): "
                  + "  ".join(f"b{s['block']}:{s['score_all_rows']:.4f}" + ("" if s["status"] == "ok" else "!") for s in res["steps"]), flush=True)
            save()
    save()
    print(f"{'rows':>5} {'block':>5} {'subsets':>7} {'mean':>8} {'std':>7} {'best':>8} {'line search':>12} {'seconds':>8}")
    for r in summarize(done):
        print(f"{r['rows']:5d} {r['block']:5d} {r['subsets']:7d} {r['mean']:8.4f} {r['std']:7.4f} {r['best']:8.4f} "
              f"{r.get('line_search_mean', float('nan')):12.4f} {r['seconds_mean']:8.0f}")
    print(f"DQI {INSTANCE['dqi']:.4f}; wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
