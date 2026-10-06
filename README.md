# dqi-screening-code

Code and data for the paper **"Discovering New Problems for Decoded Quantum Interferometry"**
(R. Mazumder and P. Niroula).

Decoded quantum interferometry (DQI) turns a decoder for an error-correcting code into an
approximation algorithm for max-LINSAT: given a matrix `B` over a finite field and an accepted set
(or a target) for each row, find `x` so that as many rows of `Bx` as possible are satisfied. The
paper derives rules that a problem must pass before DQI can beat simple classical baselines, and
studies two new candidate problems: alternant max-agreement (AMA), posed on a McEliece-type public
key, and multiplicative polynomial intersection (MPI). Optimal polynomial intersection (OPI) is the
reference problem. The scripts' `--family` options for the three are `opi`, `alternant` and
`logcauchy`.

This repository contains the scripts behind the paper's numbers, the small library they use, and
the stored outputs of the long runs.

## Quick start

You need Python 3.11 or later.

```bash
git clone https://github.com/magic-state-labs/dqi-screening-code
cd dqi-screening-code
python -m pip install -r requirements.txt
python scripts/48_paper_derived_numbers.py
```

The last command takes a few seconds. It recomputes the paper's closed-form numbers, reads the
stored results of the long runs, and prints each value next to the value in the paper (Tables 2,
3, 5 and 8 and the numbers quoted in the text, except the row "MPI, h = 17" of Table 2, for which see
`REPRODUCE.md`; the scaling tables 9-13 are printed by script 52):

```
...
(2) Table 2: Q_Pr, best classical (range over seeds), exact finite DQI, classical - DQI
  OPI, m = 400: Q_Pr                      0.548878               paper: 0.549                ok
  OPI, m = 400: best classical            0.6700 to 0.6725       paper: 0.6700 to 0.6725     ok
  OPI, m = 400: DQI (finite)              0.685604               paper: 0.6856               ok
  ...
173 values, 0 differ
```

The same list is written to `results/paper-derived-numbers.json`, which git ignores.

Run all commands from the repository root. Nothing needs to be installed beyond
`requirements.txt` (numpy, OR-Tools, psutil), except for the lattice attacks, which also need
`fpylll` (`python -m pip install fpylll cysignals`, Linux or macOS), and the scaling figures, which need
`matplotlib` (version 3.11.2 reproduces the stored image files byte for byte).

## What is here

| path | contents |
|---|---|
| `scripts/` | one numbered script per analysis; "script 48" in the paper is `scripts/48_...py` |
| `src/dqi_explorer/` | the library the scripts import (instance constructions, decoders, the DQI quality formulas, the classical solvers) |
| `runs/` | small wrappers used for the long runs |
| `results/` | stored outputs of the long runs (JSON) and their console logs; `results/scaling/` holds the runs with budgets counted in moves |
| `REPRODUCE.md` | the commands behind the tables, following the supplement's table of scripts |

## Reproducing the tables

| in the paper | command | time |
|---|---|---|
| Tables 2 (without the h = 17 row), 3, 5, 8 and the numbers in the text, from stored results | `python scripts/48_paper_derived_numbers.py` | seconds |
| the rules evaluated on the 109 catalog entries | `python scripts/10_advantage_checklist_sweep.py` | under 2 s |
| Table of finite-length comparisons: OPI rows | `python scripts/18_opi_finite_control.py --sizes 4001 --tag s4` | 5 min |
| same table: AMA rows | `python runs/run_ext6.py 30 14000 3101` | 25-40 min, 4-14 GB (less with `--cpsat-workers 2`) |
| same table: MPI rows | `python scripts/41_log_cauchy_finite_control.py --sizes 2000 --seeds 3101 --line-seconds 600 --heat-seconds 600 --tag bud600` | 25 min |
| Table of lattice attacks | `python scripts/43_lattice_attack.py results/lattice/logcauchy-m600-s3101.json --subset-rows 100,150 --blocks 20 --seconds 300` | 10 min, needs fpylll; a shorter variant of the stored run |
| attack-cost estimates for the McEliece keys | `python scripts/49_gijs_cost_estimate.py` | 1 min |
| check of the general-objective law | `python scripts/44_cosine_law_certificate.py` | seconds |
| appendix on the scaling of the classical attacks: tables and figures, from stored results | `python scripts/52_scaling_figures.py` | 3 min, needs matplotlib |
| one of its runs (line search and annealing with the budget in moves) | `python scripts/50_budget_scaling.py --family logcauchy --size 600 --seed 3101 --line-moves 10000 --anneal-moves 100,1000,10000 --tag test` | 15 s on 8 cores; delete the new `-test.json` file afterwards |

`REPRODUCE.md` lists the commands for all seeds and sizes and for the remaining checks.

Each script that produces a stored result overwrites the corresponding file in `results/`;
script 48 then reports the changed values as differences. `git checkout results` restores the
stored copies.

## What to expect from a rerun

- Instances are generated from their seeds, and the DQI values are exact formulas. These reproduce
  up to floating-point rounding in the last digits.
- The classical solvers of the main comparison table (line search, simulated annealing, CP-SAT,
  lattice reduction) stop after a fixed wall-clock time, so their best scores depend on the
  machine. The scaling runs of scripts 50 and 53 count moves instead, so the machine changes
  their running time but not their budget. A rerun of scripts 50 and 53 reproduces the stored
  OPI and AMA scores exactly. For MPI the scores of individual chains can differ between
  machines (the cosine scores are floating-point numbers), and only their statistics reproduce. A faster machine does more
  search in the same time and reaches higher scores. The table shows reruns of the documented
  commands on a faster machine (Apple M5 Max) next to the stored values used in the paper; all
  remain on the same side of DQI's value.

  | run (seed 3101 unless stated) | stored | rerun | DQI |
  |---|---|---|---|
  | OPI, m = 4000, seeds 2101-2103 | 0.5850 to 0.5900 | 0.5880 to 0.5900 | 0.7109 |
  | AMA, m = 14000 | 0.5144 | 0.5161 | 0.5321 |
  | MPI, m = 2000, 600 s | 0.1493 | 0.1696 | 0.2433 |
  | MPI, m = 6000, 40 s | 0.1128 | 0.1169 | 0.2518 |
  | lattice, m = 600, BKZ-20 on 150 rows | 0.2873 | 0.2669 | 0.2235 |

- The stored AMA runs and five MPI runs (`m = 2000`, the 1800 s run at `m = 6000`, and `h = 17`)
  include the best assignment each solver found. `python scripts/48_paper_derived_numbers.py --witnesses`
  rebuilds each instance from its seed and re-scores these assignments (about an hour).

## Notes

- The paper's rules are (R1)-(R4). Most scripts print them under earlier labels: (D0)-(D3) are the
  parts of (R1), (D4) is (R2), (D5) is (R3), (D6) is (R4), and (D7) is the objective-dependent form
  of (R1) and (R2). Script 10 numbers its screening conditions C0-C8 and lists them in the first
  part of its output.
- Scripts 30 and 36 default to extension degree `s = 4`. The paper uses `s = 6` and
  `alpha = 0.08`; `runs/run_ext6.py` passes these options.
- The stored fixed-time runs were made with Python 3.13 on a 16 GB laptop (Windows 11), with the lattice runs
  under Linux; the runs in `results/scaling/` were made on 80-core machines. Console logs of the long runs
  are in `results/logs-bud600/` and `results/lattice/`.
  Scripts 50, 51 and 53 need Linux or macOS.

## Citation

```bibtex
@misc{mazumder2026discovering,
  title  = {Discovering New Problems for Decoded Quantum Interferometry},
  author = {Mazumder, Rhik and Niroula, Pradeep},
  year   = {2026}
}
```

## License

MIT, see `LICENSE`.
