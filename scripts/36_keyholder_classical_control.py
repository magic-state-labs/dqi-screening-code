"""Script 36 - A classical solver WITH the key, on the alternant (McEliece) instance.

Script 30 attacks the public matrix without the key.  This script asks whether a classical key
holder can do better.  To the key holder the instance is trace-OPI over ``F_q``, ``q = p^s``: the
optimizer code is ``{ (Tr(v_i g(a_i)))_i : deg g < r }``, so the solution space is the ``r``
coefficients of ``g`` over ``F_q``.  The key gives the ability to move along ``F_q``-lines
``g + t w(z)``, ``w = prod_{l in L}(z - a_l)``, which pin the ``r - 1`` rows in ``L`` and
re-randomize the other ``m - r + 1`` rows over the ``q`` candidates ``t``; the key-less line
search of script 21 moves along ``F_p``-lines that pin ``n - 1 = s r - 1`` rows and offer
``p = 11`` candidates.  Attacks:

  KA  Prange over F_q: interpolate g from r rows given accepted trace values (a calibration);
  KB  F_q pinned-line search: L = r - 1 currently satisfied rows, best move taken, restart after
      a number of moves without improvement;
  KC  heat-bath version of KB (t drawn with probability exp(beta * count), beta annealed);
  KE  KB and KC started from the best assignment of the key-less portfolio (script 30);
  KD  Guruswami--Sudan list-recovery threshold, printed as vacuous.

When ``q`` exceeds ``--full-line-max-q`` (default 2x10^5) each move scores ``t = 0`` and
``--line-samples`` (default 4096) random points of the line, each exactly, instead of all ``q``.

Every assignment is mapped back to ``y = Tr(v g(a))`` in public order, checked to lie in
``im(B)`` and re-scored.  The key-less numbers and the finite DQI value are read from the
script 30 records for the same seeds.

The paper's runs (s = 6, m = 14000 and 20000) are started through the wrapper:
  python runs/run_ext6.py 36 14000 3101
Stored runs are printed by:
  python scripts/36_keyholder_classical_control.py --merge
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXAMPLES)
EXPLORER = os.path.join(ROOT, "src")
sys.path.insert(0, EXPLORER)

from dqi_explorer.alternant_trapdoor import as_dense_arrays, build_alternant_trapdoor_instance  # noqa: E402
from dqi_explorer.discovery import prange_quality  # noqa: E402
from dqi_explorer.spectral import canonical_finite_quality  # noqa: E402


def import_module_quietly(name: str):
    if EXAMPLES not in sys.path:
        sys.path.insert(0, EXAMPLES)
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        return __import__(name)
    finally:
        sys.argv = saved


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


P, S = 11, int(arg("--ext", "4"))
KEYLESS_TAG = arg("--keyless-tag", "bud600")
ACCEPTED = (P - 1) // 2
QUICK = "--quick" in sys.argv
SIZES = [int(v) for v in arg("--sizes", "6000").split(",")]
SEEDS = [int(v) for v in arg("--seeds", "3101,3102,3103").split(",")]
if QUICK:
    SEEDS = SEEDS[:1]
ALPHA = float(arg("--alpha", "0.12"))
T_LINE = float(arg("--seconds", "5" if QUICK else "40"))
T_LONG = float(arg("--long-seconds", "0" if QUICK else "300"))
T_WARM = float(arg("--warm-seconds", "5" if QUICK else "600"))
LINE_SAMPLES = int(arg("--line-samples", "4096"))  # candidates per F_q-line when q is large
FULL_LINE_MAX_Q = int(arg("--full-line-max-q", "200000"))  # scan every t on the line up to this q
T_HEAT = T_LINE
PRANGE_Q = int(arg("--prange-q", "50" if QUICK else "300"))
RESULTS_DIR = os.path.join(ROOT, "results")
TAG = arg("--tag", "quick" if QUICK else "")


class TraceOPI:
    """The key holder's view: y_i = Tr(v_i g(a_i)) in public order, g over F_q of degree < r."""

    def __init__(self, inst, ctl, rng):
        self.inst, self.ctl, self.rng = inst, ctl, rng
        self.field = inst.field
        self.q = self.field.order
        self.r = inst.r
        self.m = inst.m
        perm = inst.permutation
        self.a = np.asarray(inst.points, dtype=np.int64)[perm]  # public order
        self.v = np.asarray(inst.multipliers, dtype=np.int64)[perm]
        # trace table over F_q
        elements = np.arange(self.q, dtype=np.int64)
        total = elements.copy()
        power = elements.copy()
        for _ in range(1, S):
            power = self.field.pow(power, P)
            total = self.field.add(total, power)
        assert np.all(self.field.is_subfield(total))
        self.TR = self.field.DIGITS[total][:, 0]  # trace as an element of F_p
        # hyperplanes T_u = {x : Tr(x) = u}, stored as logs (zero excluded, only in T_0)
        self.hyper_logs = [self.field.LOG[np.nonzero((self.TR == u) & (elements != 0))[0]] for u in range(P)]
        self.ar = np.arange(self.m)

    def evaluate(self, g) -> np.ndarray:
        """y_i = Tr(v_i g(a_i)) for coefficient vector g (low -> high)."""

        value = np.zeros(self.m, dtype=np.int64)
        for coefficient in reversed(np.asarray(g, dtype=np.int64).tolist()):
            value = self.field.add(self.field.mul(value, self.a), coefficient)
        return self.TR[self.field.mul(self.v, value)]

    def score(self, y) -> int:
        return int(self.ctl.M[self.ar, y].sum())

    def interpolate(self, rows, values) -> np.ndarray:
        """g of degree < r with g(a_rows) = values (r rows), by solving the Vandermonde system."""

        from dqi_explorer.alternant_trapdoor import solve_square_over_field

        pts = self.a[rows]
        V = np.zeros((self.r, self.r), dtype=np.int64)
        col = np.ones(self.r, dtype=np.int64)
        for j in range(self.r):
            V[:, j] = col
            col = self.field.mul(col, pts)
        return solve_square_over_field(self.field, V, np.asarray(values, dtype=np.int64))

    def random_g(self):
        return self.rng.integers(0, self.q, self.r, dtype=np.int64)

    # KA: Prange over F_q
    def prange_q(self, restarts) -> tuple[dict, np.ndarray]:
        best, best_g, scores = -1, None, []
        for _ in range(restarts):
            rows = self.rng.choice(self.m, self.r, replace=False)
            values = np.zeros(self.r, dtype=np.int64)
            for k, i in enumerate(rows):
                u = int(self.rng.choice(self.ctl.A[i]))  # accepted trace value
                x = int(self.field.EXP[self.rng.choice(self.hyper_logs[u])]) if u else int(self.rng.choice(np.nonzero(self.TR == 0)[0]))
                values[k] = self.field.mul(x, self.field.inv(self.v[i]))
            g = self.interpolate(rows, values)
            if g is None:
                continue
            y = self.evaluate(g)
            s = self.score(y)
            scores.append(s)
            if s > best:
                best, best_g = s, g
        return {"best": best, "restarts": len(scores), "mean": float(np.mean(scores)) / self.m}, best_g

    # the histogram over t for the line g + t w
    def line_counts(self, y, c, pinned_mask, chunk=800) -> np.ndarray:
        counts = np.zeros(self.q, dtype=np.int64)
        free = np.nonzero(~pinned_mask)[0]
        c_free = c[free]
        assert np.all(c_free != 0)
        log_cinv = (-self.field.LOG[c_free]) % (self.q - 1)
        A = self.ctl.A  # m x ACCEPTED accepted values
        for start in range(0, len(free), chunk):
            rows = free[start : start + chunk]
            lc = log_cinv[start : start + chunk]
            targets = (A[rows] - y[rows][:, None]) % P  # rows x ACCEPTED: needed Tr(c t)
            idx_parts = []
            for u in range(P):
                sel = np.nonzero(targets == u)
                if len(sel[0]) == 0:
                    continue
                logs = (self.hyper_logs[u][None, :] + lc[sel[0]][:, None]) % (self.q - 1)
                idx_parts.append(self.field.EXP[logs].ravel())
                if u == 0:  # t = 0 lies in T_0 too
                    counts[0] += len(sel[0])
            if idx_parts:
                counts += np.bincount(np.concatenate(idx_parts), minlength=self.q)
        counts += int(self.ctl.M[np.nonzero(pinned_mask)[0], y[pinned_mask]].sum())
        return counts

    def line_counts_sampled(self, y, c, k, chunk=256):
        """Exact scores of y + Tr(c t) for t = 0 and k random nonzero t (large q: a full scan is m q work)."""

        cand = np.concatenate([[0], self.rng.integers(1, self.q, size=k)]).astype(np.int64)
        nz = c != 0
        log_c = self.field.LOG[c[nz]]
        counts = np.empty(len(cand), dtype=np.int64)
        counts[0] = self.score(y)
        for start in range(1, len(cand), chunk):
            ts = cand[start : start + chunk]
            yy = np.repeat(y[:, None], len(ts), axis=1)
            prod = self.field.EXP[(log_c[:, None] + self.field.LOG[ts][None, :]) % (self.q - 1)]
            yy[nz] = (yy[nz] + self.TR[prod]) % P
            counts[start : start + len(ts)] = self.ctl.M[self.ar[:, None], yy].sum(axis=0)
        return cand, counts

    def line_vector(self, pinned_rows) -> np.ndarray:
        """c_i = v_i w(a_i), w = prod_{l in L}(z - a_l)."""

        w = np.ones(self.m, dtype=np.int64)
        for l in pinned_rows:
            w = self.field.mul(w, self.field.sub(self.a, self.a[l]))
        return self.field.mul(self.v, w)

    def w_coefficients(self, pinned_rows) -> np.ndarray:
        coeffs = np.zeros(self.r, dtype=np.int64)
        coeffs[0] = 1
        deg = 0
        for l in pinned_rows:
            new = np.zeros(self.r, dtype=np.int64)
            new[1 : deg + 2] = coeffs[: deg + 1]
            new[: deg + 1] = self.field.sub(new[: deg + 1], self.field.mul(self.a[l], coeffs[: deg + 1]))
            coeffs, deg = new, deg + 1
        return coeffs

    def g_from_codeword(self, y, harness) -> np.ndarray:
        """Coefficients g with Tr(v_i g(a_i)) = y_i, for a codeword y of im(B) in public order.

        y is F_p-linear in the s*r digits of g, so evaluate the s*r unit digit vectors, pick an
        invertible set of s*r rows and solve over F_p.
        """

        unit = {}
        for x in range(self.q):
            d = self.field.DIGITS[x]
            if int(np.count_nonzero(d)) == 1 and int(d[np.nonzero(d)[0][0]]) == 1:
                unit[int(np.nonzero(d)[0][0])] = x
        cols, basis = [], []
        for j in range(self.r):
            for k in range(S):
                g = np.zeros(self.r, dtype=np.int64)
                g[j] = unit[k]
                basis.append(g)
                cols.append(self.evaluate(g))
        Mmat = np.stack(cols, axis=1)  # m x (s r) over F_p
        fp = harness.PrimeField(P)
        for _ in range(50):
            rows = np.sort(self.rng.choice(self.m, Mmat.shape[1], replace=False))
            inv = harness.gauss_inverse(fp, Mmat[rows])
            if inv is not None:
                break
        else:
            raise RuntimeError("no invertible row set")
        coeff = fp.matmul(inv, np.asarray(y, dtype=np.int64)[rows][:, None])[:, 0]
        g = np.zeros(self.r, dtype=np.int64)
        for c, b in zip(coeff.tolist(), basis):
            for _ in range(int(c)):  # c in F_p: adding b c times multiplies it by c
                g = self.field.add(g, b)
        assert np.array_equal(self.evaluate(g), np.asarray(y, dtype=np.int64)), "codeword not reproduced"
        return g

    # KB: F_q pinned-line search
    def line_search(self, seconds, patience=30, heat=False, beta0=0.05, beta1=4.0, g0=None) -> tuple[dict, np.ndarray]:
        t_end = time.time() + seconds
        t0 = time.time()
        best, best_g, moves, restarts = -1, None, 0, 0
        g = self.random_g() if g0 is None else np.asarray(g0, dtype=np.int64).copy()
        y = self.evaluate(g)
        cur, stale = self.score(y), 0
        restarts += 1
        while time.time() < t_end:
            sat = np.nonzero(self.ctl.M[self.ar, y])[0]
            if len(sat) < self.r - 1:
                pinned = self.rng.choice(self.m, self.r - 1, replace=False)
            else:
                pinned = self.rng.choice(sat, self.r - 1, replace=False)
            mask = np.zeros(self.m, dtype=bool)
            mask[pinned] = True
            c = self.line_vector(pinned)
            if self.q > FULL_LINE_MAX_Q:
                cand, counts = self.line_counts_sampled(y, c, LINE_SAMPLES)
            else:
                cand, counts = None, self.line_counts(y, c, mask)
            assert counts[0] == cur, (counts[0], cur)
            moves += 1
            if heat:
                frac = (time.time() - t0) / seconds
                beta = beta0 * (beta1 / beta0) ** min(frac, 1.0)
                wts = np.exp(beta * (counts - counts.max()))
                j = int(self.rng.choice(len(counts), p=wts / wts.sum()))
            else:
                top = int(counts.max())
                j = int(self.rng.choice(np.nonzero(counts == top)[0])) if top > cur else 0
            t = int(cand[j]) if cand is not None else j
            if t:
                g = self.field.add(g, self.field.mul(t, self.w_coefficients(pinned)))
                y = self.evaluate(g)
                new = self.score(y)
                assert new == counts[j], (new, counts[j])
                stale = 0 if new > cur else stale + 1
                cur = new
            else:
                stale += 1
            if cur > best:
                best, best_g = cur, g.copy()
            if not heat and stale >= patience:
                g = self.random_g() if g0 is None else best_g.copy()
                y = self.evaluate(g)
                cur, stale = self.score(y), 0
                restarts += 1
        return {"best": best, "moves": moves, "restarts": restarts, "seconds": seconds}, best_g


def keyless_witness(size: int, seed: int):
    """The best stored codeword of the key-less 600 s portfolio (script 30 --tag bud600), if any."""

    path = os.path.join(RESULTS_DIR, f"finite-control-alternant-run-{KEYLESS_TAG}-m{size}-s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rec = json.load(handle)["results"][0]
    attacks = {k: v for k, v in rec["attacks"].items() if isinstance(v, dict) and "witness_y" in v and "best" in v}
    if not attacks:
        return None
    best = max(attacks.values(), key=lambda v: v["best"])
    return np.asarray(best["witness_y"], dtype=np.int64)


def keyless_record(size: int, seed: int) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"finite-control-alternant-run-m{size}-s{seed}.json")
    if S != 4:
        path = os.path.join(RESULTS_DIR, f"finite-control-alternant-run-{KEYLESS_TAG}-m{size}-s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["results"][0]


def run_one(harness, size: int, seed: int) -> dict:
    t0 = time.time()
    inst = build_alternant_trapdoor_instance(m=size, seed=seed, p=P, s=S, alpha=ALPHA)
    m, n = inst.m, inst.n
    B, sets = as_dense_arrays(inst)
    field = harness.PrimeField(P)
    rng = np.random.default_rng(seed + 36)
    ctl = harness.Control(field, B, 0, sets, rng)
    view = TraceOPI(inst, ctl, rng)
    quality = canonical_finite_quality(m, P, ACCEPTED, inst.ell)
    dqi_count = quality.value * m
    q_pr = prange_quality(n / m, ACCEPTED / P)
    print(f"alternant size={m} seed={seed}: r={inst.r} n={n} q={view.q} | DQI {dqi_count:.1f} ({quality.value:.4f}) Q_Pr {q_pr:.4f}", flush=True)
    # sanity: a random g maps into im(B)
    y0 = view.evaluate(view.random_g())
    ctl.verify(y0)
    record = {
        "family": "alternant-keyholder",
        "m": m,
        "n": n,
        "r": inst.r,
        "seed": seed,
        "q": view.q,
        "dqi_lemma92": dqi_count,
        "Q_fin": quality.value,
        "Q_Pr": q_pr,
        "attacks": {},
    }
    schedule = [
        ("KA_prange_q", lambda: view.prange_q(PRANGE_Q)),
        ("KB_line_q", lambda: view.line_search(T_LINE)),
        ("KC_heat_q", lambda: view.line_search(T_HEAT, heat=True)),
    ]
    if T_LONG:
        schedule.append(("KB_line_q_long", lambda: view.line_search(T_LONG)))
    warm = keyless_witness(size, seed)
    if warm is not None and T_WARM:
        g_warm = view.g_from_codeword(warm, harness)
        start = view.score(view.evaluate(g_warm))
        print(f"   KE warm start from the key-less portfolio's best codeword: {start} ({start / m:.4f})", flush=True)
        schedule.append(("KE_line_q_warm", lambda: view.line_search(T_WARM, g0=g_warm)))
        schedule.append(("KE_heat_q_warm", lambda: view.line_search(T_WARM, heat=True, g0=g_warm)))
    for key, call in schedule:
        res, g = call()
        y = view.evaluate(g)
        independent = ctl.verify(y)
        assert independent == res["best"], (key, independent, res["best"])
        res["fraction"] = res["best"] / m
        record["attacks"][key] = res
        print(f"   {key}: {res['best']} ({res['fraction']:.4f}) {({k: v for k, v in res.items() if k not in ('best', 'fraction')})}", flush=True)
        res["witness_y"] = [int(v) for v in np.asarray(y).ravel()]  # the public-order codeword, re-scorable
    # KD: Guruswami-Sudan list recovery over F_q with lists of size rho q: needs agreement > sqrt(r m L)
    L = ACCEPTED * P ** (S - 1)
    record["KD_list_recovery_agreement_needed"] = float(np.sqrt(inst.r * m * L))
    print(f"   KD list recovery: needs agreement > sqrt(r m L) = {record['KD_list_recovery_agreement_needed']:.0f} > m = {m}: vacuous", flush=True)
    keyless = keyless_record(size, seed)
    if keyless:
        record["keyless"] = {k: v.get("fraction") for k, v in keyless["attacks"].items() if isinstance(v, dict) and "fraction" in v}
        record["keyless_best"] = keyless["best_classical"] / m
        print(f"   key-less (script 30): {record['keyless']} best {record['keyless_best']:.4f}", flush=True)
    scored = {k: v["best"] for k, v in record["attacks"].items()}
    best_key = max(scored, key=scored.get)
    record["best_keyholder"] = scored[best_key]
    record["best_keyholder_fraction"] = scored[best_key] / m
    record["best_attack"] = best_key
    record["keyholder_minus_dqi"] = scored[best_key] - dqi_count
    record["beats_dqi"] = bool(scored[best_key] >= dqi_count)
    record["seconds"] = time.time() - t0
    print(
        f"   best key-holder {scored[best_key]} ({scored[best_key] / m:.4f}) by {best_key} | DQI {dqi_count:.1f} | "
        f"diff {scored[best_key] - dqi_count:+.1f} | {'BEATEN' if record['beats_dqi'] else 'below DQI'} | {record['seconds']:.0f}s",
        flush=True,
    )
    return record


def output_path(size: int, seed: int) -> str:
    if TAG:
        return os.path.join(RESULTS_DIR, f"finite-control-alternant-keyholder-run-{TAG}-m{size}-s{seed}.json")
    return os.path.join(RESULTS_DIR, f"finite-control-alternant-keyholder-run-m{size}-s{seed}.json")


def main() -> None:
    harness = import_module_quietly("21_finite_control_harness")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for size in SIZES:
        for seed in SEEDS:
            record = run_one(harness, size, seed)
            payload = {"family": "alternant-keyholder", "p": P, "s": S, "alpha": ALPHA,
                       "budgets_seconds": {"line": T_LINE, "heat": T_HEAT, "long": T_LONG}, "results": [record]}
            with open(output_path(size, seed), "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.write("\n")
            print("   wrote", os.path.relpath(output_path(size, seed), ROOT), flush=True)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "finite-control-alternant-keyholder-run-*m*.json"))):
            with open(path, encoding="utf-8") as handle:
                rec = json.load(handle)["results"][0]
            attacks = "  ".join(f"{k}:{v['fraction']:.4f}" for k, v in rec["attacks"].items())
            print(f"m={rec['m']} seed={rec['seed']} DQI {rec['Q_fin']:.4f} key-less best {rec.get('keyless_best', float('nan')):.4f} | {attacks} | best key-holder {rec['best_keyholder_fraction']:.4f} ({'beaten' if rec['beats_dqi'] else 'below DQI'})")
    else:
        main()
