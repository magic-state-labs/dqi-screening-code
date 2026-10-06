"""Script 50 - Budget scaling of the classical portfolio, with budgets counted in moves.

The portfolio runs of scripts 18, 30 and 41 stop after a fixed wall-clock time, so their scores
depend on the machine.  This script runs the same two local searches with the budget counted in
moves, which is machine independent, and records the best score as a function of the number of
moves:

  line search   the pivoting line search (restart after 60 moves without improvement), run for
                ``--line-moves`` moves per chain; the best score so far is recorded at a grid of
                move counts;
  annealing     the heat-bath search, one run for each length N in ``--anneal-moves``; the inverse
                temperature rises geometrically from 0.05 to 4 over the N moves.

One move scores all q points of one codeword line on all m rows, then updates the systematic
basis (about m n field operations).  The scoring takes about m r operations for accepted sets of
size r, and m operations plus one FFT of length q for the cosine.  ``--chains`` independent
chains are run for each instance (``--jobs`` at a time).  For annealing the script records, for
each run length N, the best score visited by each chain; for the line search, the best score of
each chain against the number of moves.

Families (the instances of the paper, regenerated from their seeds):
  opi         optimal polynomial intersection, prime p = m + 1, n = round(m/10)   (seeds 2101-2103)
  alternant   McEliece-type alternant key, p = 11, s = 6, alpha = 0.08             (seeds 3101-3103)
  logcauchy   multiplicative polynomial intersection, M = 8191, alpha = 0.07       (seeds 3101-3103)

Run from the repository root:
  python scripts/50_budget_scaling.py --family logcauchy --size 2000 --seed 3101 --chains 8 --jobs 8 \
      --line-moves 20000 --anneal-moves 300,1000,3000,10000
Writes results/scaling/budget-<family>-m<m>-s<seed>[-tag].json.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per chain; the chains are the parallelism

import argparse
import json
import multiprocessing
import sys
import time
from math import sqrt

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dqi_explorer.screen import Control, CosineObjective, PrimeField, SetObjective  # noqa: E402
from dqi_explorer.spectral import canonical_finite_quality  # noqa: E402

PATIENCE, BETA0, BETA1 = 60, 0.05, 4.0


# ---------------------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------------------
def cosine_finite(m: int, ell: int) -> float:
    A = np.zeros((ell + 1, ell + 1))
    for k in range(1, ell + 1):
        A[k - 1, k] = A[k, k - 1] = sqrt(k * (m - k + 1))
    return float(np.linalg.eigvalsh(A)[-1]) / (sqrt(2) * m)


def build(family: str, size: int, seed: int) -> dict:
    """The instance as plain arrays, with DQI's exact finite value and the Prange baseline."""

    if family == "opi":
        p = size  # a prime; m = p - 1 evaluation points, as in script 18
        m, n, r = p - 1, round((p - 1) / 10), (p - 1) // 2
        rng = np.random.default_rng(seed)
        pts = np.arange(1, p, dtype=np.int64)
        B = np.ones((m, n), dtype=np.int64)
        for j in range(1, n):
            B[:, j] = B[:, j - 1] * pts % p
        sets = [rng.choice(p, r, replace=False).astype(np.int64) for _ in range(m)]
        ell = (n - 1) // 2
        return {"p": p, "B": B, "sets": sets, "centres": None, "ell": ell,
                "dqi": canonical_finite_quality(m, p, r, ell).value, "prange": r / p + (1 - r / p) * n / m}
    if family == "alternant":
        from dqi_explorer.alternant_trapdoor import as_dense_arrays, build_alternant_trapdoor_instance

        inst = build_alternant_trapdoor_instance(m=size, seed=seed, p=11, s=6, alpha=0.08)
        B, sets = as_dense_arrays(inst)
        return {"p": 11, "B": B, "sets": sets, "centres": None, "ell": inst.ell,
                "dqi": canonical_finite_quality(inst.m, 11, 5, inst.ell).value, "prange": 5 / 11 + (6 / 11) * inst.n / inst.m}
    if family == "logcauchy":
        from dqi_explorer.log_cauchy import build_log_cauchy_instance

        inst = build_log_cauchy_instance(m=size, seed=seed)
        return {"p": inst.p, "B": np.asarray(inst.B_pub, dtype=np.int64), "sets": None,
                "centres": np.asarray(inst.centres, dtype=np.int64), "ell": inst.ell,
                "dqi": cosine_finite(inst.m, inst.ell), "prange": inst.n / inst.m}
    raise SystemExit("unknown family " + family)


def control(inst: dict, rng) -> Control:
    objective = SetObjective(inst["sets"], inst["p"]) if inst["sets"] is not None else CosineObjective(inst["centres"], inst["p"])
    return Control(PrimeField(inst["p"]), inst["B"], objective, rng)


# ---------------------------------------------------------------------------------------
# the two searches, with the budget in moves (same moves as Control.line and Control.heat)
# ---------------------------------------------------------------------------------------
def grid(limit: int) -> list[int]:
    """1, 2, 3, 4, 6, 8, 12, 16, ... up to the limit, and the limit itself."""
    out, k = set(), 1
    while k <= limit:
        out.update(v for v in (k, k + k // 2) if v <= limit)
        k *= 2
    out.add(limit)
    return sorted(out)


def compact(L: np.ndarray, p: int) -> np.ndarray:
    """The systematic basis in the smallest integer type that holds a product of two field elements."""
    bound = (p - 1) ** 2
    return L.astype(np.int8 if bound <= 127 else np.int16 if bound <= 32767 else np.int32 if bound < 2 ** 31 else np.int64)


def inverse_mod(a: np.ndarray, p: int, field) -> np.ndarray | None:
    """Inverse of a square matrix over F_p by Gauss-Jordan elimination on the compact type (None if singular)."""
    n = a.shape[0]
    aug = compact(np.concatenate([np.asarray(a, dtype=np.int64) % p, np.eye(n, dtype=np.int64)], axis=1), p)
    for c in range(n):
        nz = np.nonzero(aug[c:, c])[0]
        if len(nz) == 0:
            return None
        piv = c + int(nz[0])
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        row = ((aug[c, c:].astype(np.int64) * int(field.inv(int(aug[c, c])))) % p).astype(aug.dtype)
        aug[c, c:] = row
        f = aug[:, c].copy()
        f[c] = 0
        rows = np.nonzero(f)[0]
        if len(rows):
            X = f[rows, None] * row[None, :]
            np.subtract(aug[rows, c:], X, out=X)
            X %= p
            aug[rows, c:] = X
    return aug[:, n:].astype(np.int64)


def start(ctl: Control):
    """Control._start with the same random choices and the faster inverse."""
    while True:
        S = ctl.rng.choice(ctl.m, ctl.n, replace=False)
        inv = inverse_mod(ctl.B[S], ctl.q, ctl.f)
        if inv is not None:
            break
    L = ctl.f.matmul(ctl.B, inv)
    t = ctl.obj.start_target(S, ctl.rng)
    return S, compact(L, ctl.q), ctl.f.matmul(L, t[:, None])[:, 0]


def pivot(ctl: Control, S, L, k, satisfied):
    """Control._pivot with the same random choices, on the compact basis (several times faster)."""
    cand = np.setdiff1d(np.nonzero(satisfied & (L[:, k] != 0))[0], S)
    if len(cand) == 0:
        return S, L
    j, p = int(ctl.rng.choice(cand)), ctl.q
    piv_inv = int(ctl.f.inv(int(L[j, k])))
    rowj = ((L[j, :].astype(np.int64) * piv_inv) % p).astype(L.dtype)
    colk = L[:, k].copy()
    Lnew = colk[:, None] * rowj[None, :]
    np.subtract(L, Lnew, out=Lnew)
    Lnew %= p
    Lnew[:, k] = ((colk.astype(np.int64) * piv_inv) % p).astype(L.dtype)
    S = S.copy()
    S[k] = j
    return S, Lnew


def line_search(ctl: Control, moves_total: int) -> list[tuple[int, float]]:
    obj, marks, curve = ctl.obj, grid(moves_total), []
    best, moves = None, 0
    while moves < moves_total:
        S, L, y = start(ctl)
        cur, stale = obj.score(y), 0
        while stale < PATIENCE and moves < moves_total:
            k = int(ctl.rng.integers(ctl.n))
            u = L[:, k].astype(np.int64)
            counts = obj.line_scores(y, u, ctl.f)
            top = counts.max()
            moves += 1
            if obj.better(top, cur):
                choices = np.nonzero(counts >= top - (0 if obj.is_integer else 1e-9))[0]
                t = int(ctl.rng.choice(choices))
                y = ctl.f.add(y, ctl.f.mul(t, u))
                cur, stale = (int(top) if obj.is_integer else float(counts[t])), 0
            else:
                stale += 1
            if best is None or obj.better(cur, best):
                best = cur
            while marks and marks[0] <= moves:
                curve.append((marks.pop(0), float(best)))
            S, L = pivot(ctl, S, L, k, obj.satisfied(y))
    return curve


def anneal(ctl: Control, moves_total: int) -> float:
    obj = ctl.obj
    S, L, y = start(ctl)
    best = obj.score(y)
    for step in range(moves_total):
        beta = BETA0 * (BETA1 / BETA0) ** (step / moves_total)
        k = int(ctl.rng.integers(ctl.n))
        u = L[:, k].astype(np.int64)
        counts = obj.line_scores(y, u, ctl.f)
        w = np.exp(beta * (counts - counts.max()))
        t = int(ctl.rng.choice(ctl.q, p=w / w.sum()))
        y = ctl.f.add(y, ctl.f.mul(t, u))
        cur = int(counts[t]) if obj.is_integer else float(counts[t])
        if obj.better(cur, best):
            best = cur
        S, L = pivot(ctl, S, L, k, obj.satisfied(y))
    return float(best)


INSTANCE: dict = {}


def run_task(task: tuple) -> dict:
    kind, chain, moves, seed = task
    ctl = control(INSTANCE, np.random.default_rng([seed, chain, moves, 0 if kind == "line" else 1]))
    t0 = time.time()
    if kind == "line":
        return {"kind": kind, "chain": chain, "moves": moves, "curve": line_search(ctl, moves), "seconds": time.time() - t0}
    return {"kind": kind, "chain": chain, "moves": moves, "best": anneal(ctl, moves), "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------
def summarize(inst: dict, tasks_done: list[dict], chains: int) -> dict:
    m = inst["B"].shape[0]
    lines = [t for t in tasks_done if t["kind"] == "line"]
    line_curve = []
    if lines:
        for i, (mv, _) in enumerate(lines[0]["curve"]):
            vals = [t["curve"][i][1] / m for t in lines if i < len(t["curve"])]
            line_curve.append({"moves_per_chain": mv, "total_moves": mv * len(vals), "best": max(vals), "mean": float(np.mean(vals)), "chains": len(vals)})
    anneal_curve = []
    for N in sorted({t["moves"] for t in tasks_done if t["kind"] == "anneal"}):
        vals = [t["best"] / m for t in tasks_done if t["kind"] == "anneal" and t["moves"] == N]
        anneal_curve.append({"moves": N, "total_moves": N * len(vals), "best": max(vals), "mean": float(np.mean(vals)),
                             "std": float(np.std(vals)), "chains": len(vals)})
    return {"line_search": line_curve, "annealing": anneal_curve}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=("opi", "alternant", "logcauchy"))
    parser.add_argument("--size", type=int, required=True, help="m (for opi: the prime p = m + 1)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--line-moves", type=int, default=2000, help="moves per line-search chain (0: skip)")
    parser.add_argument("--anneal-moves", default="100,300,1000", help="annealing lengths, comma separated (empty: skip)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    global INSTANCE
    t0 = time.time()
    INSTANCE = build(args.family, args.size, args.seed)
    m, n = INSTANCE["B"].shape
    print(f"{args.family} m={m} n={n} q={INSTANCE['p']} seed={args.seed}: DQI {INSTANCE['dqi']:.4f}, Prange {INSTANCE['prange']:.4f} (built in {time.time() - t0:.0f}s)", flush=True)
    lengths = [int(v) for v in args.anneal_moves.split(",") if v]
    tasks = [("anneal", c, N, args.seed) for N in sorted(lengths, reverse=True) for c in range(args.chains)]
    if args.line_moves:
        tasks = [("line", c, args.line_moves, args.seed) for c in range(args.chains)] + tasks
    out_dir = os.path.join(ROOT, "results", "scaling")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"budget-{args.family}-m{m}-s{args.seed}{'-' + args.tag if args.tag else ''}.json")
    done: list[dict] = []

    def save() -> None:
        payload = {"family": args.family, "m": m, "n": n, "q": INSTANCE["p"], "seed": args.seed, "ell": INSTANCE["ell"],
                   "dqi_finite": INSTANCE["dqi"], "prange": INSTANCE["prange"], "chains": args.chains,
                   "patience": PATIENCE, "beta": [BETA0, BETA1],
                   "summary": summarize(INSTANCE, done, args.chains), "runs": done}
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=1)
            fh.write("\n")
        os.replace(path + ".tmp", path)

    last = time.time()
    with multiprocessing.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(run_task, tasks, chunksize=1):
            done.append(res)
            score = (res["curve"][-1][1] if res["kind"] == "line" else res["best"]) / m
            print(f"   {res['kind']:6s} chain {res['chain']:3d} moves {res['moves']:8d}: {score:.4f}  ({res['seconds']:.0f}s, {res['moves'] / max(res['seconds'], 1e-9):.1f} moves/s)", flush=True)
            if time.time() - last > 60:
                save()
                last = time.time()
    save()
    s = summarize(INSTANCE, done, args.chains)
    if s["line_search"]:
        e = s["line_search"][-1]
        print(f"line search: best {e['best']:.4f} after {e['total_moves']} moves in total")
    for e in s["annealing"]:
        print(f"annealing N={e['moves']}: mean {e['mean']:.4f} +- {e['std']:.4f}, best of {e['chains']} {e['best']:.4f}")
    print(f"DQI {INSTANCE['dqi']:.4f}; wrote {os.path.relpath(path, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
