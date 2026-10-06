"""Script 30 - Classical portfolio against finite DQI for alternant (McEliece) max-agreement, without the key.

The family: ``B = P H_p^T S`` for a scrambled alternant parity check over the prime field
``F_11`` with extension degree ``--ext`` (default 4; the paper uses 6 through
``runs/run_ext6.py``), accepted sets of size 5 per row.  The key holder decodes ``ker(B^T)`` with Berlekamp--Massey to
``ell = floor((r-1)/2)`` errors, a worst-case licence; the classical controls below see only
``(B, accepted sets)``.

Per (size, seed):
  1. build the instance and certify ``rank(H_p) = s r`` (``dqi_explorer.alternant_trapdoor``);
  2. verify the key holder's decoder on random patterns of weight <= ell (exact recovery);
  3. model DQI with the finite Lemma 9.2 value ``canonical_finite_quality(m, 11, 5, ell)``,
     cross-checked against the harness's own tridiagonal, the Prange baseline
     ``rho + (1 - rho) alpha`` and the restart-converse threshold
     ``(1 - alpha) ln(N/eta) / (2 mu^2)`` at ``N = 2^40``, ``eta = 0.05``;
  4. run the generic attacks of ``21_finite_control_harness.py`` on the public matrix:
     A information-set restarts (Prange), B pivoting line search, C heat bath, and
     E an anytime CP-SAT model adapted from ``18_opi_finite_control.py`` (full model up to
     ``--cpsat-max-m`` rows, otherwise a random row subset, hinted with the best heuristic
     witness and re-scored on all rows).  Every witness is re-scored independently by
     ``Control.verify`` (membership in the code plus a pure-Python count).
  5. write the record to a JSON file under ``results/``.

Verdict rule (``--merge``): a size "survives the generic portfolio" iff the best classical score
is below the finite DQI value on every seed; otherwise "beaten on the instance".  Nothing here is
an advantage claim: the hardness route for this family is conditional (key indistinguishability,
now known to be at most superpolynomial by the syzygy distinguisher 2407.15740, plus
Prange-optimality on random dense max-LINSAT), and these controls are generic attacks with fixed
budgets.

Run from the repository root:
  python scripts/30_alternant_trapdoor_finite_control.py --sizes 2000 --quick
  python scripts/30_alternant_trapdoor_finite_control.py --sizes 6000
  python scripts/30_alternant_trapdoor_finite_control.py --merge
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from math import log

import numpy as np

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXAMPLES)
EXPLORER = os.path.join(ROOT, "src")
sys.path.insert(0, EXPLORER)

from dqi_explorer.alternant_trapdoor import (  # noqa: E402
    as_dense_arrays,
    build_alternant_trapdoor_instance,
    square_code_note,
    verify_decoder,
)
from dqi_explorer.discovery import prange_quality  # noqa: E402
from dqi_explorer.spectral import canonical_finite_quality  # noqa: E402


def import_harness():
    """Import 21_finite_control_harness.py without letting it parse our command line."""

    if EXAMPLES not in sys.path:
        sys.path.insert(0, EXAMPLES)
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        return __import__("21_finite_control_harness")
    finally:
        sys.argv = saved


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


P = 11
S = int(arg("--ext", "4"))  # extension degree; s >= 6 keeps the 2026 GRS-subcode distinguisher above 2^80
ACCEPTED = (P - 1) // 2
RESTARTS = 2**40
FAILURE = 0.05
MAX_LENGTH = P**S - 1

QUICK = "--quick" in sys.argv
SIZES = [int(v) for v in arg("--sizes", "2000,6000").split(",")]
SEEDS = [int(v) for v in arg("--seeds", "3101,3102,3103").split(",")]
if QUICK:
    SEEDS = SEEDS[:1]
ALPHA = float(arg("--alpha", "0.12"))
T_LINE = float(arg("--line-seconds", "5" if QUICK else "40"))
T_HEAT = float(arg("--heat-seconds", "5" if QUICK else "40"))
T_CPSAT = float(arg("--cpsat-seconds", "10" if QUICK else "90"))
CPSAT_MAX_M = int(arg("--cpsat-max-m", "3000"))
CPSAT_WORKERS = int(arg("--cpsat-workers", "8"))
NO_CPSAT = "--no-cpsat" in sys.argv
DECODER_TRIALS = int(arg("--decoder-trials", "20" if QUICK else "200"))
SQUARE_RANK = "--square-rank" in sys.argv
# accepted sets: "random" (uniform 5-subsets, the family of the paper) or "interval"
# (a_i, a_i+1, ..., a_i+4 mod 11 for a uniform a_i: the ISIS-infinity form of the problem,
# which adds a lattice attack surface)
SETS = arg("--sets", "random")
# --random-matrix: replace the key by a uniformly random matrix of the same shape (Assumption (B))
RANDOM_MATRIX = "--random-matrix" in sys.argv
if SETS not in ("random", "interval"):
    raise SystemExit("--sets must be random or interval")
TAG = arg("--tag", "quick" if QUICK else "")
RESULTS_DIR = os.path.join(ROOT, "results")


def hoeffding_length(alpha: float, margin: float, draws: int, failure: float) -> float | None:
    """Restart converse: best of N restarts is below Q*m past this m."""

    if margin <= 0:
        return None
    return (1 - alpha) * log(draws / failure) / (2 * margin**2)


def coefficients_for(harness, ctl, y: np.ndarray) -> np.ndarray:
    """Coefficients x with B x = y, through a random invertible information set."""

    for _ in range(20):
        info = np.sort(ctl.rng.choice(ctl.m, ctl.n, replace=False))
        inv = harness.gauss_inverse(ctl.f, ctl.B[info])
        if inv is not None:
            break
    else:
        raise RuntimeError("no invertible information set found")
    x = ctl.f.matmul(inv, y[info][:, None])[:, 0]
    assert np.array_equal(ctl.f.matmul(ctl.B, x[:, None])[:, 0], y), "hint is not in the code"
    return x


def attack_cpsat(harness, ctl, hint_y: np.ndarray, seconds: float, max_rows: int, workers: int):
    """Anytime CP-SAT on the exact accepted-set model (adapted from 18_opi_finite_control.py)."""

    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # a broken OR-tools install must not kill the control
        return {"skipped": f"unavailable: {type(exc).__name__}"}, None
    try:
        return _attack_cpsat(cp_model, harness, ctl, hint_y, seconds, max_rows, workers)
    except Exception as exc:  # record, never kill the control
        return {"skipped": f"failed: {type(exc).__name__}: {exc}"[:300]}, None


def _attack_cpsat(cp_model, harness, ctl, hint_y: np.ndarray, seconds: float, max_rows: int, workers: int):
    p, m, n = ctl.q, ctl.m, ctl.n
    subset = m > max_rows
    rows = np.sort(ctl.rng.choice(m, max_rows, replace=False)) if subset else np.arange(m)
    t_build = time.time()
    model = cp_model.CpModel()
    coef = [model.NewIntVar(0, p - 1, f"c{j}") for j in range(n)]
    hits = []
    B_list = ctl.B.tolist()
    # complete hint: coefficients, every row's word and quotient, and the hit indicators
    hint = coefficients_for(harness, ctl, hint_y)
    hint_list = hint.tolist()
    for variable, value in zip(coef, hint_list):
        model.AddHint(variable, int(value))
    for i in rows.tolist():
        weights = B_list[i]
        word = model.NewIntVar(0, p - 1, f"w{i}")
        quotient = model.NewIntVar(0, sum(weights) * (p - 1) // p, f"t{i}")
        try:
            expression = cp_model.LinearExpr.WeightedSum(coef, weights)
        except AttributeError:  # older OR-tools
            expression = sum(w * c for w, c in zip(weights, coef) if w)
        model.Add(expression == word + p * quotient)
        hit = model.NewBoolVar(f"b{i}")
        model.AddAllowedAssignments([word], [[int(v)] for v in sorted(ctl.sets[i])]).OnlyEnforceIf(hit)
        hits.append(hit)
        total = sum(w * c for w, c in zip(weights, hint_list))
        model.AddHint(word, int(hint_y[i]))
        model.AddHint(quotient, (total - int(hint_y[i])) // p)
        model.AddHint(hit, int(int(hint_y[i]) in ctl.sets[i]))
    model.Maximize(sum(hits))
    build_seconds = time.time() - t_build
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 7
    status = solver.StatusName(solver.Solve(model))
    solver_score = None
    if status in ("OPTIMAL", "FEASIBLE"):
        c = np.array([solver.Value(v) for v in coef], dtype=np.int64)
        y = ctl.f.matmul(ctl.B, c[:, None])[:, 0]
        subset_hits = int(solver.ObjectiveValue())
        bound = int(solver.BestObjectiveBound())
        solver_score = ctl.score(y)
        if solver_score < ctl.score(hint_y):  # the hint is itself a feasible witness
            y = hint_y.copy()
    else:
        y = hint_y.copy()
        subset_hits = None
        bound = None
    result = {
        "status": status,
        "best": ctl.score(y),
        "solver_witness_score": solver_score,
        "hint_score": ctl.score(hint_y),
        "bound": bound,
        "subset": bool(subset),
        "rows_modelled": int(len(rows)),
        "subset_hits": subset_hits,
        "wall": float(solver.WallTime()),
        "build_seconds": build_seconds,
        "workers": int(workers),
        "seconds_limit": float(seconds),
    }
    return result, y


def interval_sets(m: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Accepted set of row i is {a_i, ..., a_i + ACCEPTED - 1} mod P for a uniform start a_i."""

    rng = np.random.default_rng(seed + 7)
    starts = rng.integers(0, P, m)
    sets = [np.array(sorted(int((a + j) % P) for j in range(ACCEPTED)), dtype=np.int64) for a in starts]
    return sets, starts


def attack_babai(ctl, starts: np.ndarray) -> tuple[dict, np.ndarray]:
    """Least-squares rounding toward the interval centres (a lattice-flavoured heuristic).

    The interval form asks for (Bx)_i mod p close to c_i = a_i + 2.  A proper CVP attack would
    reduce the q-ary lattice {Bx + p k}; at m = 2000-6000 that is out of reach here, so this
    solves the real least-squares system B x ~ c + p k for a few random integer lifts k, rounds
    x to F_p^n, and keeps the best score.  It is labelled as rounding without LLL in the record;
    a negative result here is not evidence about reduced-lattice attacks.
    """

    p, m, n = ctl.q, ctl.m, ctl.n
    centres = (starts + ACCEPTED // 2).astype(np.float64)
    B = ctl.B.astype(np.float64)
    best, best_y, trials = -1, None, 0
    try:
        import fpylll  # noqa: F401

        lll_available = True
    except Exception:
        lll_available = False
    pinv = np.linalg.pinv(B)
    for trial in range(32):
        lift = ctl.rng.integers(0, 3, m) if trial else np.zeros(m)
        target = centres + p * lift
        x = np.rint(pinv @ target).astype(np.int64) % p
        y = ctl.f.matmul(ctl.B, x[:, None])[:, 0]
        score = ctl.score(y)
        trials += 1
        if score > best:
            best, best_y = score, y.copy()
    return {"best": int(best), "trials": trials, "lll_used": False, "lll_available": lll_available,
            "note": "least-squares rounding toward interval centres; no lattice reduction at this dimension"}, best_y


def square_rank_check(harness, ctl, rng) -> dict:
    """Rank of m + 100 random Schur products of columns of B (expected m for an unstructured square)."""

    count = ctl.m + 100
    i = rng.integers(0, ctl.n, count)
    j = rng.integers(0, ctl.n, count)
    products = (ctl.B[:, i] * ctl.B[:, j]) % ctl.q
    return {"products": int(count), "rank": int(harness.rank_mod(ctl.f, products)), "m": int(ctl.m)}


def run_one(harness, size: int, seed: int) -> dict:
    t0 = time.time()
    if size > MAX_LENGTH:
        print(f"   size {size} clamped to {MAX_LENGTH} (evaluation points are nonzero)", flush=True)
        size = MAX_LENGTH
    inst = build_alternant_trapdoor_instance(m=size, seed=seed, p=P, s=S, alpha=ALPHA)
    m, n = inst.m, inst.n
    print(
        f"alternant size={m} seed={seed}: p={P} s={S} r={inst.r} n={n} modulus={inst.modulus} "
        f"attempts={inst.build_attempts} ell={inst.ell}",
        flush=True,
    )
    trials = DECODER_TRIALS if m <= 6000 or "--decoder-trials" in sys.argv else min(DECODER_TRIALS, 50)
    decoder = verify_decoder(inst, trials, seed + 1)
    print(
        f"   decoder: {decoder['exact_recoveries']}/{decoder['trials']} exact "
        f"(failures {decoder['failures']}); beyond radius {decoder['beyond_radius']}; "
        f"{decoder['seconds']:.1f}s",
        flush=True,
    )

    B, sets = as_dense_arrays(inst)
    if RANDOM_MATRIX:
        B = np.random.default_rng(seed + 7_000_000).integers(0, P, size=B.shape, dtype=np.int64)
        print("   matrix: uniformly random over F_%d, same shape %s (Assumption (B) control)" % (P, B.shape), flush=True)
    starts = None
    if SETS == "interval":
        sets, starts = interval_sets(m, seed)
    field = harness.PrimeField(P)
    rng = np.random.default_rng(seed)
    ctl = harness.Control(field, B, 0, sets, rng)
    # rank(B) = n is certified by any 2n rows of rank n (a submatrix cannot exceed the full rank, and
    # n is the column count); the full elimination at m = 20000 needs several 250 MB temporaries
    rank_rng = np.random.default_rng(seed + 9_000_000)  # separate stream: the attacks' draws are unchanged
    rows = np.sort(rank_rng.choice(m, 2 * n, replace=False)) if m > 2 * n else np.arange(m)
    rank = int(harness.rank_mod(field, B[rows]))
    if rank != n:
        rank = int(harness.rank_mod(field, B))
    assert rank == n, f"rank {rank} != {n}"
    alpha = rank / m
    ell = inst.ell
    quality = canonical_finite_quality(m, P, ACCEPTED, ell)
    dqi_count = quality.value * m
    cross = harness.dqi_expected(m, ell, P, ACCEPTED)
    assert abs(dqi_count - cross) < 1e-6, (dqi_count, cross)
    rho = ACCEPTED / P
    q_pr = prange_quality(alpha, rho)
    mu_fin = quality.value - q_pr
    threshold = hoeffding_length(alpha, mu_fin, RESTARTS, FAILURE)
    print(
        f"   model: rank {rank} alpha {alpha:.4f} rho {rho:.4f} ell {ell} | Q_fin {quality.value:.4f} "
        f"({dqi_count:.1f}) Q_Pr {q_pr:.4f} mu_fin {mu_fin:+.4f} | threshold {threshold:.0f} "
        f"({'above' if m >= threshold else 'below'}) | semicircle {harness.semicircle(ell / m, rho):.4f}",
        flush=True,
    )

    record = {
        "family": "alternant",
        "size": size,
        "seed": seed,
        "p": P,
        "s": S,
        "q": P,
        "extension_order": P**S,
        "modulus": list(inst.modulus),
        "m": m,
        "r": inst.r,
        "n": n,
        "rank": rank,
        "alpha": alpha,
        "ell": ell,
        "designed_distance": inst.designed_distance,
        "accepted_size": ACCEPTED,
        "sets": SETS,
        "rho": rho,
        "genus": 0,
        "Q_fin": {"value": quality.value, "lower": quality.lower, "upper": quality.upper},
        "dqi_lemma92": dqi_count,
        "dqi_crosscheck_abs_diff": abs(dqi_count - cross),
        "semicircle": harness.semicircle(ell / m, rho),
        "Q_Pr": q_pr,
        "mu_fin": mu_fin,
        "threshold": threshold,
        "above_threshold": bool(m >= threshold),
        "decoder": decoder,
        "structure": square_code_note(inst),
        "info": f"alternant p={P} s={S} r={inst.r}",
        "matrix": "uniformly random (Assumption (B) control)" if RANDOM_MATRIX else "alternant key",
        "attacks": {},
    }

    budget = int(min(8000, max(1000, 4e9 / (m * n))))
    best_y = None
    for key, call in (
        ("A_prange", lambda: ctl.prange(budget)),
        ("B_line", lambda: ctl.line(T_LINE)),
        ("C_heat", lambda: ctl.heat(T_HEAT)),
    ):
        res, y = call()
        independent = ctl.verify(y)
        assert independent == res["best"], (key, independent, res["best"])
        res["fraction"] = res["best"] / m
        res["witness_y"] = [int(v) for v in np.asarray(y).ravel()]  # the codeword Bx, re-scorable
        record["attacks"][key] = res
        if best_y is None or res["best"] > ctl.score(best_y):
            best_y = y.copy()
        print(f"   {key}: {res['best']} ({res['fraction']:.4f})", flush=True)
    record["attacks"]["A_prange"]["mean_minus_Q_Pr"] = record["attacks"]["A_prange"]["mean"] - q_pr

    if NO_CPSAT:
        record["attacks"]["E_cpsat"] = {"skipped": "disabled"}
    else:
        res, y = attack_cpsat(harness, ctl, best_y, T_CPSAT, CPSAT_MAX_M, CPSAT_WORKERS)
        if y is not None:
            independent = ctl.verify(y)
            assert independent == res["best"], ("E_cpsat", independent, res["best"])
            res["fraction"] = res["best"] / m
            res["witness_y"] = [int(v) for v in np.asarray(y).ravel()]
            print(
                f"   E_cpsat: {res['best']} ({res['fraction']:.4f}) status {res['status']} "
                f"rows {res['rows_modelled']} build {res['build_seconds']:.0f}s solve {res['wall']:.0f}s",
                flush=True,
            )
        else:
            print(f"   E_cpsat: {res['skipped']}", flush=True)
        record["attacks"]["E_cpsat"] = res

    if SETS == "interval":
        res, y = attack_babai(ctl, starts)
        independent = ctl.verify(y)
        assert independent == res["best"], ("F_babai", independent, res["best"])
        res["fraction"] = res["best"] / m
        record["attacks"]["F_babai"] = res
        print(f"   F_babai: {res['best']} ({res['fraction']:.4f}) [{res['note']}]", flush=True)

    if SQUARE_RANK and m <= 2000:
        record["structure"]["square_rank_check"] = square_rank_check(harness, ctl, rng)
        print(f"   square rank check: {record['structure']['square_rank_check']}", flush=True)

    scored = {k: v for k, v in record["attacks"].items() if "best" in v}
    best_key = max(scored, key=lambda k: scored[k]["best"])
    best = scored[best_key]["best"]
    record["best_attack"] = best_key
    record["best_classical"] = int(best)
    record["classical_minus_dqi"] = best - dqi_count
    record["classical_minus_dqi_fraction"] = (best - dqi_count) / m
    record["beats_dqi"] = bool(best >= dqi_count)
    record["seconds"] = time.time() - t0
    record["budgets"] = {
        "prange_restarts": budget,
        "line": T_LINE,
        "heat": T_HEAT,
        "cpsat": None if NO_CPSAT else T_CPSAT,
        "cpsat_rows": CPSAT_MAX_M,
    }
    print(
        f"   best classical {best} ({best / m:.4f}) by {best_key} | DQI {dqi_count:.1f} ({quality.value:.4f}) | "
        f"best - DQI = {best - dqi_count:+.1f} | {'BEATEN' if best >= dqi_count else 'below DQI'} | "
        f"{record['seconds']:.0f}s",
        flush=True,
    )
    return record


def output_path(size: int, seed: int) -> str:
    suffix = ("" if SETS == "random" else f"-{SETS}") + ("-randommatrix" if RANDOM_MATRIX else "")
    if TAG:
        return os.path.join(RESULTS_DIR, f"finite-control-alternant-run-{TAG}-m{size}-s{seed}{suffix}.json")
    return os.path.join(RESULTS_DIR, f"finite-control-alternant-run-m{size}-s{seed}{suffix}.json")


def write(path: str, results: list[dict]) -> None:
    payload = {
        "family": "alternant",
        "p": P,
        "s": S,
        "alpha": ALPHA,
        "accepted_size": ACCEPTED,
        "sets": SETS,
        "budgets_seconds": {"line": T_LINE, "heat": T_HEAT, "cpsat": None if NO_CPSAT else T_CPSAT},
        "results": results,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")


def run() -> None:
    harness = import_harness()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for size in SIZES:
        for seed in SEEDS:
            record = run_one(harness, size, seed)
            write(output_path(size, seed), [record])
            print("   wrote", os.path.relpath(output_path(size, seed), ROOT), flush=True)


def merge() -> None:
    groups: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "finite-control-alternant-run-*.json"))):
        if "-quick" in os.path.basename(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for record in json.load(handle)["results"]:
                key = (int(record["m"]), record.get("sets", "random"))
                groups.setdefault(key, []).append(record)
    if not groups:
        print("no alternant records found")
        return
    print(
        "%-6s %-5s %-5s %-6s %-4s %-7s %-7s %-8s %-9s %-5s  %-40s %-8s %-9s %s"
        % ("m", "seed", "rank", "alpha", "ell", "Q_fin", "Q_Pr", "mu_fin", "threshold", "above", "attacks (fraction)", "best", "best-DQI", "seconds")
    )
    for key in sorted(groups):
        m, sets_mode = key
        print(f"  [sets = {sets_mode}]")
        for rec in sorted(groups[key], key=lambda r: r["seed"]):
            attacks = "  ".join(
                f"{k.split('_')[0]}:{v['fraction']:.4f}" for k, v in rec["attacks"].items() if "fraction" in v
            )
            print(
                "%-6d %-5d %-5d %-6.4f %-4d %-7.4f %-7.4f %-+8.4f %-9.0f %-5s  %-40s %-8d %-+9.1f %.0f"
                % (
                    m,
                    rec["seed"],
                    rec["rank"],
                    rec["alpha"],
                    rec["ell"],
                    rec["Q_fin"]["value"],
                    rec["Q_Pr"],
                    rec["mu_fin"],
                    rec["threshold"],
                    "yes" if rec["above_threshold"] else "no",
                    attacks,
                    rec["best_classical"],
                    rec["classical_minus_dqi"],
                    rec["seconds"],
                )
            )
        records = groups[key]
        beaten = [r for r in records if r["beats_dqi"]]
        decoders_ok = all(r["decoder"]["all_recovered"] for r in records)
        verdict = (
            "survives the generic portfolio at this size"
            if not beaten
            else "beaten on the instance by " + ", ".join(f"seed {r['seed']} ({r['best_attack']})" for r in beaten)
        )
        print(
            f"  m={m} sets={sets_mode}: {len(records)} run(s); decoders exact on every run: {decoders_ok}; "
            f"above threshold: {all(r['above_threshold'] for r in records)}; verdict: {verdict}"
        )
        if sets_mode == "interval":
            babai = [r["attacks"].get("F_babai") for r in records if "F_babai" in r["attacks"]]
            if babai:
                heur = [max(v["best"] for k, v in r["attacks"].items() if k != "F_babai" and "best" in v) for r in records]
                print(
                    "  Babai (least-squares rounding, no LLL) vs best heuristic per seed: "
                    + ", ".join(f"{b['best']} vs {h}" for b, h in zip(babai, heur))
                )
        print(
            "  (verdict is against the generic portfolio without the key at fixed budgets; "
            "it is not a hardness statement)"
        )


if __name__ == "__main__":
    merge() if "--merge" in sys.argv else run()
