"""Script 53 - Does the key help a classical local search?  Key-holder search with the budget in moves.

Script 36 gives the classical key holder 600 s, which at s = 6 is about 180 moves.  This script
runs the same two key-holder searches for a fixed number of moves and records the best score
along the way, next to a key-less line search started from the same assignment:

  key line (warm)   pinned-line search along F_{p^s}-lines of message polynomials, started from the
                    best assignment of the stored key-less portfolio run (script 30);
  key heat (warm)   the heat-bath version, same start, inverse temperature 0.05 to 4 over the run;
  key line (cold)   pinned-line search from a random polynomial;
  keyless (warm)    the key-less pivoting line search of script 50, same start, same number of moves.

A key-holder move scores t = 0 and 4096 random points of one extension-field line, each exactly,
as in script 36.  A key-less move scores all 11 points of one prime-field line.

Run from the repository root (reads results/finite-control-alternant-run-ext6bud600-m<m>-s<seed>.json):
  python scripts/53_keyholder_scaling.py --size 14000 --seed 3101 --moves 3000 --chains 2
Writes results/scaling/keyholder-m<m>-s<seed>[-tag].json.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)


def load(name: str, argv: list[str]):
    """Import a numbered script as a module, with the options it reads at import time."""
    saved, sys.argv = sys.argv, [name] + argv
    try:
        spec = importlib.util.spec_from_file_location(name.split("_", 1)[1][:-3], os.path.join(HERE, name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


keyholder = load("36_keyholder_classical_control.py", ["--ext", "6", "--alpha", "0.08", "--keyless-tag", "ext6bud600"])
harness = load("21_finite_control_harness.py", [])
scaling = load("50_budget_scaling.py", [])

PATIENCE_KEY, BETA0, BETA1 = 30, 0.05, 4.0
STATE: dict = {}


def key_search(view, moves_total: int, heat: bool, g0) -> list[tuple[int, int]]:
    """TraceOPI.line_search of script 36 with the budget in moves; returns (moves, best score) pairs."""
    rng, marks, curve = view.rng, scaling.grid(moves_total), []
    g = view.random_g() if g0 is None else np.asarray(g0, dtype=np.int64).copy()
    y = view.evaluate(g)
    cur, stale = view.score(y), 0
    best, best_g = cur, g.copy()
    for step in range(moves_total):
        sat = np.nonzero(view.ctl.M[view.ar, y])[0]
        pinned = rng.choice(view.m if len(sat) < view.r - 1 else sat, view.r - 1, replace=False)
        cand, counts = view.line_counts_sampled(y, view.line_vector(pinned), keyholder.LINE_SAMPLES)
        if heat:
            beta = BETA0 * (BETA1 / BETA0) ** (step / moves_total)
            w = np.exp(beta * (counts - counts.max()))
            j = int(rng.choice(len(counts), p=w / w.sum()))
        else:
            top = int(counts.max())
            j = int(rng.choice(np.nonzero(counts == top)[0])) if top > cur else 0
        t = int(cand[j])
        if t:
            g = view.field.add(g, view.field.mul(t, view.w_coefficients(pinned)))
            y = view.evaluate(g)
            new = view.score(y)
            stale = 0 if new > cur else stale + 1
            cur = new
        else:
            stale += 1
        if cur > best:
            best, best_g = cur, g.copy()
        if not heat and stale >= PATIENCE_KEY:
            g = view.random_g() if g0 is None else best_g.copy()
            y = view.evaluate(g)
            cur, stale = view.score(y), 0
        while marks and marks[0] <= step + 1:
            curve.append((marks.pop(0), int(best)))
    return curve


def keyless_search(ctl, y, moves_total: int) -> list[tuple[int, int]]:
    """The pivoting line search of script 50 from the codeword y, for a fixed number of moves."""
    obj, marks, curve = ctl.obj, scaling.grid(moves_total), []
    while True:
        S = ctl.rng.choice(ctl.m, ctl.n, replace=False)
        inv = scaling.inverse_mod(ctl.B[S], ctl.q, ctl.f)
        if inv is not None:
            break
    L = scaling.compact(ctl.f.matmul(ctl.B, inv), ctl.q)
    cur = obj.score(y)
    for step in range(moves_total):
        k = int(ctl.rng.integers(ctl.n))
        u = L[:, k].astype(np.int64)
        counts = obj.line_scores(y, u, ctl.f)
        top = counts.max()
        if top > cur:
            t = int(ctl.rng.choice(np.nonzero(counts >= top)[0]))
            y = ctl.f.add(y, ctl.f.mul(t, u))
            cur = int(top)
        S, L = scaling.pivot(ctl, S, L, k, obj.satisfied(y))
        while marks and marks[0] <= step + 1:
            curve.append((marks.pop(0), int(cur)))
    return curve


def run_task(task: tuple) -> dict:
    kind, chain = task
    inst, B, sets, warm, moves, seed = (STATE[k] for k in ("inst", "B", "sets", "warm", "moves", "seed"))
    rng = np.random.default_rng([seed, chain, ("key_line_warm", "key_heat_warm", "key_line_cold", "keyless_warm").index(kind)])
    t0 = time.time()
    if kind == "keyless_warm":
        ctl = scaling.Control(scaling.PrimeField(11), B, scaling.SetObjective(sets, 11), rng)
        curve = keyless_search(ctl, warm.copy(), moves)
    else:
        view = keyholder.TraceOPI(inst, harness.Control(harness.PrimeField(11), B, 0, sets, rng), rng)
        g0 = None if kind == "key_line_cold" else view.g_from_codeword(warm, harness)
        curve = key_search(view, moves, kind == "key_heat_warm", g0)
    return {"kind": kind, "chain": chain, "curve": curve, "seconds": time.time() - t0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=14000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--moves", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    from dqi_explorer.alternant_trapdoor import as_dense_arrays, build_alternant_trapdoor_instance
    from dqi_explorer.spectral import canonical_finite_quality

    inst = build_alternant_trapdoor_instance(m=args.size, seed=args.seed, p=11, s=6, alpha=0.08)
    B, sets = as_dense_arrays(inst)
    warm = keyholder.keyless_witness(args.size, args.seed)
    if warm is None:
        raise SystemExit("no stored key-less run with assignments for this size and seed (script 30)")
    dqi = canonical_finite_quality(inst.m, 11, 5, inst.ell).value
    start = int(scaling.SetObjective(sets, 11).score(warm))
    STATE.update(inst=inst, B=B, sets=sets, warm=warm, moves=args.moves, seed=args.seed)
    print(f"alternant m={inst.m} seed={args.seed}: DQI {dqi:.4f}; warm start {start} ({start / inst.m:.4f})", flush=True)
    tasks = [(kind, c) for c in range(args.chains) for kind in ("key_line_warm", "key_heat_warm", "key_line_cold", "keyless_warm")]
    done = []
    with multiprocessing.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(run_task, tasks, chunksize=1):
            done.append(res)
            print(f"   {res['kind']:14s} chain {res['chain']}: best {res['curve'][-1][1]} ({res['curve'][-1][1] / inst.m:.4f}) after {args.moves} moves, "
                  f"{res['seconds']:.0f}s ({args.moves / res['seconds']:.2f} moves/s)", flush=True)
    out_dir = os.path.join(ROOT, "results", "scaling")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"keyholder-m{inst.m}-s{args.seed}{'-' + args.tag if args.tag else ''}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"m": inst.m, "seed": args.seed, "dqi_finite": dqi, "warm_start": start, "moves": args.moves,
                   "line_samples": keyholder.LINE_SAMPLES, "runs": done}, fh, indent=1)
        fh.write("\n")
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
