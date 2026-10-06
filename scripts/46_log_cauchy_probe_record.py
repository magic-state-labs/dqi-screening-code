"""Script 46 - Decoder probes for multiplicative polynomial intersection, stored as JSON.

Script 40 (``40_log_cauchy_screen.py``, section B) verifies the multiplicative-polynomial-
intersection decoder on built instances and prints the result.  This script stores the same checks:

  * at ``m = 2000``: the 62 restricted patterns of weight at most ``ell`` decoded exactly, the six
    all-plus patterns at ``ell + 1`` rejected, and the in-measure recovery fractions at the probe
    weights (``0.74 n`` and ``0.99 n``);
  * at ``m = 600``: the Schur square of ``im(B)`` against a random-matrix control.

This script rebuilds the same instances (``seed = 40``, ``h = 13``), runs the same calls with the
same seeds, and writes one JSON under ``results/``.  It also replays the pattern schedule
of ``verify_decoder`` with the same generator so that every pattern is recorded with its weight and
whether it sits at the radius, which the aggregate counts do not say: of the 62 patterns, the
all-plus and all-minus cases and the even-numbered random trials are at weight ``ell``, and the
odd-numbered random trials are at a uniform weight in ``[0, ell]``.

Run from the repository root:
  python scripts/46_log_cauchy_probe_record.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqi_explorer.hidden_keys import _square_rank, random_matrix_battery  # noqa: E402
from dqi_explorer.log_cauchy import (  # noqa: E402
    build_log_cauchy_instance,
    in_measure_probe,
    random_restricted_pattern,
    verify_decoder,
)
from dqi_explorer.screen import write_draft  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "decoder-probe-logcauchy-s40.json")

BUILD_SEED = 40
H = 13
SIZES = ((600, 0.10), (2000, 0.07))
DECODER_TRIALS = 60
DECODER_SEED = 1
BEYOND_TRIALS = 6
PROBE_TRIALS = 12
PROBE_SEED = 2


def replay_schedule(inst, trials: int, seed: int) -> list[dict]:
    """The pattern schedule of ``verify_decoder`` with the same generator, one record per pattern."""

    rng = np.random.default_rng(seed)
    schedule = [("plus", inst.ell), ("minus", inst.ell)]
    for trial in range(trials):
        schedule.append(("random", inst.ell if trial % 2 == 0 else int(rng.integers(0, inst.ell + 1))))
    records = []
    for signs, weight in schedule:
        y = random_restricted_pattern(inst, weight, rng, signs=signs)
        result = inst.decode(y)
        records.append({
            "signs": signs,
            "weight": int(weight),
            "at_radius": bool(weight == inst.ell),
            "exact": bool(result is not None and np.array_equal(result, y)),
            "rejected": bool(result is None),
        })
    return records


def main() -> None:
    records = []
    for m, alpha in SIZES:
        t0 = time.time()
        inst = build_log_cauchy_instance(m=m, seed=BUILD_SEED, h=H, alpha=alpha)
        build_seconds = time.time() - t0
        rng = np.random.default_rng(BUILD_SEED)
        record = {
            "m": inst.m, "n": inst.n, "ell": inst.ell, "alpha": inst.alpha, "p": inst.p, "h": H,
            "build_seed": BUILD_SEED, "build_seconds": build_seconds,
        }

        dec = verify_decoder(inst, DECODER_TRIALS, DECODER_SEED, beyond_trials=BEYOND_TRIALS)
        patterns = replay_schedule(inst, DECODER_TRIALS, DECODER_SEED)
        assert len(patterns) == dec["trials"]
        assert sum(p["exact"] for p in patterns) == dec["exact_recoveries"], "replay disagrees with verify_decoder"
        record["decoder"] = {
            "seed": DECODER_SEED,
            "trials": dec["trials"],
            "exact_recoveries": dec["exact_recoveries"],
            "at_radius": sum(p["at_radius"] for p in patterns),
            "at_radius_exact": sum(p["at_radius"] and p["exact"] for p in patterns),
            "below_radius": sum(not p["at_radius"] for p in patterns),
            "by_kind": dec["by_kind"],
            "failures": dec["failures"],
            "beyond_radius": dec["beyond_radius"],
            "seconds": dec["seconds"],
            "patterns": patterns,
        }

        weights = [inst.ell + k for k in (1, inst.ell // 2, inst.ell, int(1.5 * inst.ell)) if inst.ell + k < inst.n]
        probe = in_measure_probe(inst, weights, PROBE_TRIALS, PROBE_SEED)
        record["in_measure"] = {
            "seed": PROBE_SEED,
            "trials_per_weight": PROBE_TRIALS,
            "weights": {str(w): {"weight": int(w), "fraction_of_n": w / inst.n, "recovery_fraction": f,
                                 "successes": int(round(f * PROBE_TRIALS))}
                        for w, f in probe["recovery_fraction"].items()},
        }

        if m <= 600:
            sq = _square_rank(np.ascontiguousarray(inst.B_pub.T), inst.p, rng)
            ctl = random_matrix_battery(inst.m, inst.n, inst.p, rng, shortenings=(0,), sides=("image",))["image_shorten_0"]
            record["schur_square"] = {
                "square_rank": sq["square_rank"], "random_expectation": sq["random_expectation"], "full": sq["full"],
                "control_square_rank": ctl["square_rank"], "control_random_expectation": ctl["random_expectation"],
            }

        records.append(record)
        print("m = %d: decoder %d/%d exact (%d at radius), beyond-radius all-plus rejected %d/%d, in-measure %s%s"
              % (m, dec["exact_recoveries"], dec["trials"], record["decoder"]["at_radius"],
                 dec["beyond_radius"]["rejected"], dec["beyond_radius"]["trials"],
                 {w: v["recovery_fraction"] for w, v in record["in_measure"]["weights"].items()},
                 ", square %d/%d" % (sq["square_rank"], sq["random_expectation"]) if m <= 600 else ""))

    write_draft(OUT_PATH, records, family="logcauchy", objective="cos",
                extra={"generator": "dqi_explorer.log_cauchy:build_log_cauchy_instance"})
    print("wrote", os.path.relpath(OUT_PATH))


if __name__ == "__main__":
    main()
