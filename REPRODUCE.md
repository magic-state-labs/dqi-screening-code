# Reproducing the paper's numbers

Run everything from the repository root, after `python -m pip install -r requirements.txt`.

**The short way.** `python scripts/48_paper_derived_numbers.py` prints the numbers of Tables 2, 3,
5 and 8 and of the text next to the paper's values (a few seconds); the row "MPI, h = 17" of Table 2 is
not included, see the section on multiplicative polynomial intersection below. Closed-form values are
recomputed; values from long runs are read from `results/`.

**The long way.** The sections below follow the table "Scripts supporting the numerical analyses"
in the supplement, row by row, and give the command that regenerates each stored result. Times are
approximate and are for a recent laptop. Commands marked *(long)* run classical solvers for a fixed
wall-clock time, so their best scores vary slightly between machines and runs (see README.md).
Commands marked *(fpylll)* need `python -m pip install fpylll cysignals`.

Seeds: 2101-2103 for OPI; 3101-3103 for the two candidates; 1701-1703 for script 15; 40 for the
decoder probe of script 46.

## Reach threshold, bound on the margin, screening inequalities (scripts 13, 23, 48)

```bash
python scripts/13_margin_cap.py
python scripts/23_margin_soundness_check.py     # also loads scripts/20_passing_families_screen.py
python scripts/48_paper_derived_numbers.py      # block (1)
```

## OPI: finite DQI values and classical comparisons (scripts 18, 48)

```bash
python scripts/18_opi_finite_control.py --sizes 101,211 --tag s1    # (long) 20 min; m = 100, 210, with CP-SAT
python scripts/18_opi_finite_control.py --sizes 401,1009 --tag s2   # (long) 10 min; m = 400
python scripts/18_opi_finite_control.py --sizes 2003 --tag s3       # (long) 5 min
python scripts/18_opi_finite_control.py --sizes 4001 --tag s4       # (long) 5 min; m = 4000
python scripts/18_opi_finite_control.py --merge                     # prints the stored runs
python scripts/48_paper_derived_numbers.py                          # block (2)
```

Each run uses the three seeds 2101-2103 and writes `results/opi-finite-control-run-<tag>.json`.

## Alternant (McEliece) max-agreement: parameters, margins, rules (scripts 29, 30, 48)

```bash
python scripts/29_alternant_trapdoor_screen.py
python scripts/48_paper_derived_numbers.py      # block (3)
```

## Alternant max-agreement: decoder checks and the public-instance portfolio (scripts 30, 48)

The paper's instances have p = 11, s = 6, alpha = 0.08. `runs/run_ext6.py` passes these options to
script 30, with 600 s for each of line search and simulated annealing, and logs time and peak
memory. Scripts 30, 36 and 53 load their solvers from `scripts/21_finite_control_harness.py`.

```bash
python runs/run_ext6.py 30 14000 3101                      # (long) 25-40 min; 4-14 GB with the default 8 CP-SAT workers
python runs/run_ext6.py 30 14000 3102 --cpsat-workers 2    # (long) 50 min, 2.5 GB
python runs/run_ext6.py 30 14000 3103 --cpsat-workers 2    # (long) 35 min
python runs/run_ext6.py 30 20000 3101                      # (long) 90 min, 3.5 GB
python runs/run_ext6.py 30 20000 3102 --cpsat-workers 2    # (long) 75-115 min; same for 3103
python scripts/30_alternant_trapdoor_finite_control.py --merge    # prints the stored runs; the 1800 s run and the random-matrix control appear as extra rows for seed 3101
python scripts/48_paper_derived_numbers.py      # blocks (2), (4), (8)
```

Each run writes `results/finite-control-alternant-run-ext6bud600-m<m>-s<seed>.json` and a log
in `results/logs-bud600/`. The record includes the decoder check (50 random error patterns within
the licensed radius).

Tripled budgets on seed 3101 (the budget-sensitivity check):

```bash
python runs/run_ext6.py 30 14000 3101 --cpsat-workers 2 --line-seconds 1800 --heat-seconds 1800 --tag ext6bud1800   # (long) 75 min
```

Uniformly random matrix of the same shape at m = 20000 (the wrapper passes options in pairs, so
the switch takes a dummy value):

```bash
python runs/run_ext6.py 30 20000 3101 --cpsat-workers 2 --tag ext6rm600 --random-matrix 1   # (long) 70 min
```

## Cost estimates for the distinguisher and key recovery (script 49)

```bash
python scripts/49_gijs_cost_estimate.py            # 1 min; estimates and a small experiment over F_11
python scripts/49_gijs_cost_estimate.py --no-exp   # estimates only; rewrites the stored file without the experiment block
```

Writes `results/gijs-cost-estimate.json`.

## Classical optimization with the alternant key (script 36)

Run after the key-less runs above; the search starts from their best assignments.

```bash
python runs/run_ext6.py 36 14000 3101     # (long) 45 min per seed; same for 3102, 3103
python runs/run_ext6.py 36 20000 3101 --warm-seconds 1800 --tag ext6warm1800   # (long) 3.2 h per seed; same for 3102, 3103
python scripts/36_keyholder_classical_control.py --merge    # prints the stored runs
```

## Reed-Solomon list-recovery threshold (script 48)

```bash
python scripts/48_paper_derived_numbers.py      # block (4)
```

## Multiplicative polynomial intersection: rules, finite DQI values, portfolio (scripts 40, 41, 48)

```bash
python scripts/40_log_cauchy_screen.py          # 20 s
python scripts/41_log_cauchy_finite_control.py --sizes 2000 --seeds 3101 --line-seconds 600 --heat-seconds 600 --tag bud600   # (long) 25 min
python scripts/41_log_cauchy_finite_control.py --sizes 6000 --seeds 3101 --line-seconds 600 --heat-seconds 600 --tag bud600   # (long) 30 min
python scripts/41_log_cauchy_finite_control.py --sizes 6000 --seeds 3101                                                      # (long) 8 min, 40 s budgets
python scripts/41_log_cauchy_finite_control.py --h 17 --sizes 20000 --seeds 3101 --line-seconds 600 --heat-seconds 600 --tag h17bud600   # (long) 2.2 h; M = 2^17 - 1
python scripts/41_log_cauchy_finite_control.py --merge      # prints the stored runs
python scripts/48_paper_derived_numbers.py      # blocks (2), (3), (5)
```

Repeat with `--seeds 3102` and `--seeds 3103`. At m = 6000 the paper reports, for each seed, the
better of the 40 s and the 600 s run. For the `--h 17` run the JSON record is stored for seed 3103;
the scores of seeds 3101 and 3102 are in `results/logs-bud600/lc-h17-m20000-s<seed>.log`. Tripled
budgets at m = 6000, seed 3101:

```bash
python scripts/41_log_cauchy_finite_control.py --sizes 6000 --seeds 3101 --line-seconds 1800 --heat-seconds 1800 --tag bud1800   # (long) 65 min
```

The stored 40 s and 1800 s records at m = 6000 also hold a least-squares rounding entry (`F_rounding`, written
with `--bkz`), which the paper does not use.

## Signed-error decoding checks and code structure (scripts 40, 46, 48)

```bash
python scripts/46_log_cauchy_probe_record.py    # 5 s; writes results/decoder-probe-logcauchy-s40.json
python scripts/40_log_cauchy_screen.py
python scripts/48_paper_derived_numbers.py      # block (5)
```

## Row-subset lattice attacks (scripts 43, 48; model of the best subset: script 54)

The instances are stored under `results/lattice/`. They can be regenerated from their seeds, e.g.

```bash
python -c "import sys; sys.path.insert(0, 'src'); from dqi_explorer.log_cauchy import build_log_cauchy_instance as b, export_instance_json as e; e(b(m=600, seed=3101), 'results/lattice/logcauchy-m600-s3101.json')"
```

Attack runs *(fpylll)*, `--blocks 2` meaning LLL alone:

```bash
python scripts/43_lattice_attack.py results/lattice/logcauchy-m600-s3101.json --subset-rows 60,100,150 --blocks 2,20,40 --seconds 300     # (long) 30 min
python scripts/43_lattice_attack.py results/lattice/logcauchy-m2000-s3101.json --subset-rows 180,220,260 --blocks 2,20 --seconds 1200     # (long) 2 h
python scripts/43_lattice_attack.py results/lattice/logcauchy-m6000-s3101.json --subset-rows 470,520 --blocks 2 --seconds 1200            # (long) 1.5 h
bash runs/lattice/run_seeds.sh             # (long) seeds 3102, 3103 at m = 600 and m = 2000
bash runs/lattice/run_m6000_one.sh 3101    # (long) BKZ-20 on 520 and 700 rows at m = 6000
python scripts/48_paper_derived_numbers.py      # block (6)
```

Each run writes `<instance>.lattice.json` next to the instance. The table in the paper lists the
100- and 150-row BKZ-20 runs at m = 600 and all runs at m = 2000 and m = 6000. The shell wrappers
use the `python` on the path; set `FPYLLL_PYTHON` to use another interpreter.

The stored logs `run-m2000-s310{2,3}.log` and `run-m6000-s310{1,2,3}-bkz.log` are from runs that failed in
double precision, as reported in the paper; `run-m6000-s3101-mpfr2.log` is the completed m = 6000 run.
`run_seeds.sh` also writes logs for m = 600.

Model of the best-subset score (script 54), seconds each:

```bash
python scripts/54_lattice_babai_model.py --m 2000 --n 140 --p 8191 --tau 0.2433
python scripts/54_lattice_babai_model.py --m 600 --n 42 --p 8191 --tau 0.2235
python scripts/54_lattice_babai_model.py --m 6000 --n 420 --p 8191 --tau 0.2518
python scripts/54_lattice_babai_model.py --m 20000 --n 1400 --p 131071 --tau 0.2562
```

Writes `results/lattice/logcauchy-params-*.babai.json`.

## General-objective expectation formula (script 44)

```bash
python scripts/44_cosine_law_certificate.py     # 10 s
```

Builds small DQI states directly (m = 8, l = 2) and compares their expected objective with the
tridiagonal formula.

## Subfield examples, affine-subspace counts, equation ranks (scripts 15-17, 48)

```bash
python scripts/15_partially_stuck_finite_control.py   # 3 min; ranks 109 and 224; writes results/partially-stuck-finite-control-run.json
python scripts/16_subfield_gate.py                    # 10 s; expected numbers of contained cosets
python scripts/17_suzuki_subfield_attack.py           # 1 min
python scripts/48_paper_derived_numbers.py            # block (7)
```

## Finite-length Prange-restart thresholds (script 48)

```bash
python scripts/48_paper_derived_numbers.py      # block (3)
```

## Scaling of the classical attacks (scripts 50-53)

These runs count the search budget in moves instead of seconds (the lattice runs: in BKZ tours,
with a time limit per block size). Their outputs are stored in `results/scaling/`.

```bash
python -m pip install matplotlib==3.11.2
python scripts/52_scaling_figures.py      # 3 min; prints the tables of the scaling appendix, writes results/scaling/figures/
```

Script 52 only reads the stored outputs. The runs themselves are long; they were made on 80-core
machines (about a day each for the largest instances). Shorter versions, with fewer moves, run on a
laptop.

**Line search and annealing with budgets in moves (script 50).** One command per instance and
seed. `--size` is `m` (for OPI, the prime `p = m + 1`), `--line-moves` the moves per line-search
chain, `--anneal-moves` the run lengths of the annealing.

```bash
python scripts/50_budget_scaling.py --family alternant --size 14000 --seed 3101 --chains 8 --line-moves 30000 --anneal-moves 100,300,1000,3000,10000,30000,100000
```

| family | `--size` | seeds | `--chains` | `--line-moves` | `--anneal-moves` |
|---|---|---|---|---|---|
| `opi` | 401 | 2101-2103 | 6 | 1000000 | 100, 300, 1000, ... up to 1000000 |
| `opi` | 1009 | 2101 | 8 | 1000000 | up to 1000000 |
| `opi` | 2003 | 2101 | 8 | 300000 | up to 300000 |
| `opi` | 4001 | 2101-2103 | 8 | 100000 | up to 100000 |
| `alternant` | 2000 | 3101 | 8 | 3000000 | up to 3000000 |
| `alternant` | 6000 | 3101 | 8 | 1000000 | up to 1000000 |
| `alternant` | 14000 | 3101-3103 | 8 | 30000 | up to 100000 |
| `alternant` | 14000, `--tag long` | 3101-3103 | 8 | 100000 | 300000 |
| `alternant` | 14000, `--tag long2` | 3101 | 8 | 300000 | 300000, 1000000 |
| `alternant` | 20000 | 3101 | 8 | 30000 | up to 30000 |
| `logcauchy` | 600 | 3101-3103 | 8 | 10000000 | up to 10000000 |
| `logcauchy` | 1000 | 3101 | 8 | 3000000 | up to 3000000 |
| `logcauchy` | 2000 | 3101-3103 | 8 | 3000000 | up to 3000000 |
| `logcauchy` | 4000 | 3101 | 8 | 1000000 | up to 1000000 |
| `logcauchy` | 6000 | 3101-3103 | 8 | 1000000 | up to 1000000 |

`--jobs` sets how many chains run at once (default: all cores). A second run of the same instance
with `--tag <name>` writes a separate file; script 52 merges the files of one instance. The tables
use a run length once all of its chains have finished on every seed.

**Searches with the alternant key (script 53).** The searches start from the best assignments of
the stored script 30 runs.

```bash
python scripts/53_keyholder_scaling.py --size 14000 --seed 3101 --moves 10000 --chains 1    # (long) 4 h per seed; same for 3102, 3103
```

**Lattice attack at block sizes up to 60 (script 51)** *(fpylll)*. This script needs fpylll with
its BKZ pruning strategies and extended-precision types; `conda install -c conda-forge fpylll`
provides both.

```bash
python scripts/51_lattice_scaling.py --size 600 --seed 3101 --rows 80,100,130,160,200,260 --blocks 2,20,30,40,50,60 --subsets 16 --tours 8 --block-seconds 7200 --polish-moves 3000
python scripts/51_lattice_scaling.py --size 1000 --seed 3101 --rows 110,140,180,220,280,340 --blocks 2,20,30,40,50,60 --subsets 16 --tours 8 --block-seconds 7200 --polish-moves 3000
python scripts/51_lattice_scaling.py --size 2000 --seed 3101 --rows 220,260,300 --blocks 2,20,30,40,50,60 --subsets 12 --tours 8 --block-seconds 7200 --polish-moves 3000
python scripts/51_lattice_scaling.py --size 2000 --seed 3101 --rows 350,400,500 --blocks 2,20,30,40,50,60 --subsets 12 --tours 8 --block-seconds 7200 --polish-moves 0 --tag large
python scripts/51_lattice_scaling.py --size 2000 --seed 3102 --rows 260,300,400 --blocks 2,20,40,50,60 --subsets 8 --tours 8 --block-seconds 7200 --polish-moves 3000    # same for 3103
python scripts/51_lattice_scaling.py --size 6000 --seed 3101 --rows 480,560,660,800 --blocks 2,20,30,40,50 --subsets 8 --tours 8 --block-seconds 7200 --polish-moves 3000
```

The stored file of the third command holds 8 of the 12 subsets at 300 rows.

## The candidate search (table of grouped outcomes)

```bash
python scripts/10_advantage_checklist_sweep.py        # the rules on the 109 catalog entries; prints "rows: 109 | ..."
python scripts/32_problem_classes.py                  # every entry has a recorded disposition
python scripts/31_new_problem_mechanisms.py           # hidden structured keys, low-density objectives
python scripts/33_metric_alphabet_relaxations.py      # restricted-alphabet lattice decoders
python scripts/34_hidden_key_screen.py --quick        # hidden keys (without --quick: adds m = 2000)
python scripts/38_other_settings.py                   # hidden LDPC key and other settings
python scripts/42_domain_switch_sweep.py --quick      # alternative maps from exponents
```
