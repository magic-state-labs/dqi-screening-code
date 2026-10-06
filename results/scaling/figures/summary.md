# Scaling runs: summary

Scores are fractions of rows satisfied (OPI, AMA) or mean cosines (MPI).
z = (score - Prange) / (sigma sqrt((1 - alpha)/m)); N = q x moves is the number of codewords scored;
zhat(N) is the level reached by the best of N independent completions, N P[Z >= zhat] = 1.

## Budget scaling of the generic searches

| family | m | seeds | DQI (finite) | z of DQI | annealing: moves per run | mean (best chain) | z | zhat(N) | line search: moves in total per seed (chains) | best chain per seed | z | zhat(N) | reaches DQI | log2 N for DQI | log2 field operations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optimal polynomial intersection (OPI) | 400 | 3 | 0.6856 | 5.76 | 1e+06 | 0.6876 (0.7025) | 5.85 | 5.85 | 6e+06 (6) | 0.6900-0.6925 | 5.99 | 6.14 | yes | 28 | 36 |
| optimal polynomial intersection (OPI) | 1008 | 1 | 0.7017 | 10.18 | 1e+06 | 0.6405 (0.6448) | 6.08 | 6.00 | 8e+06 (8) | 0.6438 | 6.31 | 6.33 | no | 79 | 89 |
| optimal polynomial intersection (OPI) | 2002 | 1 | 0.7067 | 14.81 | 3e+05 | 0.6139 (0.6234) | 6.06 | 5.91 | 2.4e+06 (8) | 0.6189 | 6.52 | 6.25 | no | 163 | 174 |
| optimal polynomial intersection (OPI) | 4000 | 3 | 0.7109 | 21.47 | 1e+05 | 0.5948 (0.5990) | 5.99 | 5.85 | 8e+05 (8) | 0.5950-0.5953 | 6.03 | 6.18 | no | 338 | 349 |
| alternant max-agreement (AMA), s = 6 | 2000 | 1 | 0.5226 | 2.24 | 3e+06 | 0.5574 (0.5650) | 5.49 | 5.42 | 2.4e+07 (8) | 0.5595 | 5.69 | 5.78 | yes | 6 | 21 |
| alternant max-agreement (AMA), s = 6 | 6000 | 1 | 0.5285 | 4.92 | 1e+06 | 0.5306 (0.5347) | 5.26 | 5.22 | 8e+06 (8) | 0.5315 | 5.40 | 5.59 | yes | 21 | 39 |
| alternant max-agreement (AMA), s = 6 | 14000 | 3 | 0.5321 | 8.37 | 3e+05 | 0.5187 (0.5212) | 5.08 | 4.99 | 1e+06 (16) | 0.5189-0.5210 | 5.35 | 5.22 | no | 55 | 75 |
| alternant max-agreement (AMA), s = 6 | 20000 | 1 | 0.5329 | 10.26 | 3e+04 | 0.5142 (0.5165) | 4.73 | 4.52 | 2.4e+05 (8) | 0.5153 | 5.04 | 4.95 | no | 81 | 102 |
| multiplicative polynomial intersection (MPI) | 600 | 3 | 0.2235 | 5.51 | 1e+07 | 0.2434 (0.2539) | 6.23 | 6.68 | 8e+07 (8) | 0.2598-0.2609 | 6.84 | 6.98 | yes | 26 | 30 |
| multiplicative polynomial intersection (MPI) | 1000 | 1 | 0.2338 | 7.59 | 3e+06 | 0.2007 (0.2043) | 6.06 | 6.50 | 2.4e+07 (8) | 0.2106 | 6.52 | 6.80 | no | 46 | 50 |
| multiplicative polynomial intersection (MPI) | 2000 | 3 | 0.2433 | 11.36 | 3e+06 | 0.1580 (0.1639) | 5.77 | 6.50 | 2.4e+07 (8) | 0.1688-0.1741 | 6.65 | 6.80 | no | 98 | 104 |
| multiplicative polynomial intersection (MPI) | 4000 | 1 | 0.2493 | 16.63 | 1e+06 | 0.1274 (0.1298) | 5.33 | 6.33 | 8e+06 (8) | 0.1421 | 6.69 | 6.64 | no | 205 | 212 |
| multiplicative polynomial intersection (MPI) | 6000 | 3 | 0.2518 | 20.65 | 1e+06 | 0.1141 (0.1178) | 5.01 | 6.33 | 8e+06 (8) | 0.1257-0.1288 | 6.51 | 6.64 | no | 313 | 322 |

## Annealing on the common scale: z of the mean minus zhat(N), for each run length

- alternant max-agreement (AMA), s = 6, m = 2000: 1e+02: +0.28, 3e+02: +0.18, 1e+03: +0.11, 3e+03: +0.01, 1e+04: +0.07, 3e+04: +0.08, 1e+05: +0.12, 3e+05: +0.03, 1e+06: -0.02, 3e+06: +0.08
- alternant max-agreement (AMA), s = 6, m = 6000: 1e+02: -0.05, 3e+02: +0.11, 1e+03: +0.13, 3e+03: +0.02, 1e+04: +0.12, 3e+04: -0.09, 1e+05: +0.10, 3e+05: +0.30, 1e+06: +0.04
- alternant max-agreement (AMA), s = 6, m = 14000: 1e+02: +0.06, 3e+02: +0.09, 1e+03: +0.09, 3e+03: +0.08, 1e+04: +0.05, 3e+04: +0.12, 1e+05: +0.08, 3e+05: +0.09
- alternant max-agreement (AMA), s = 6, m = 20000: 1e+02: -0.02, 3e+02: +0.07, 1e+03: +0.28, 3e+03: +0.06, 1e+04: -0.03, 3e+04: +0.20
- multiplicative polynomial intersection (MPI), m = 600: 1e+02: -0.39, 3e+02: -0.37, 1e+03: -0.41, 3e+03: -0.39, 1e+04: -0.40, 3e+04: -0.43, 1e+05: -0.42, 3e+05: -0.41, 1e+06: -0.40, 3e+06: -0.42, 1e+07: -0.45
- multiplicative polynomial intersection (MPI), m = 1000: 1e+02: -0.22, 3e+02: -0.43, 1e+03: -0.52, 3e+03: -0.40, 1e+04: -0.52, 3e+04: -0.64, 1e+05: -0.47, 3e+05: -0.59, 1e+06: -0.41, 3e+06: -0.44
- multiplicative polynomial intersection (MPI), m = 2000: 1e+02: -0.28, 3e+02: -0.36, 1e+03: -0.54, 3e+03: -0.67, 1e+04: -0.73, 3e+04: -0.69, 1e+05: -0.80, 3e+05: -0.73, 1e+06: -0.69, 3e+06: -0.72
- multiplicative polynomial intersection (MPI), m = 4000: 1e+02: -0.20, 3e+02: -0.26, 1e+03: -0.53, 3e+03: -0.60, 1e+04: -0.72, 3e+04: -1.00, 1e+05: -0.97, 3e+05: -1.09, 1e+06: -1.00
- multiplicative polynomial intersection (MPI), m = 6000: 1e+02: -0.02, 3e+02: -0.27, 1e+03: -0.52, 3e+03: -0.67, 1e+04: -0.88, 3e+04: -1.04, 1e+05: -1.07, 3e+05: -1.25, 1e+06: -1.32
- optimal polynomial intersection (OPI), m = 400: 1e+02: -0.16, 3e+02: -0.00, 1e+03: -0.03, 3e+03: -0.01, 1e+04: -0.03, 3e+04: -0.06, 1e+05: -0.01, 3e+05: +0.00, 1e+06: +0.00
- optimal polynomial intersection (OPI), m = 1008: 1e+02: +0.01, 3e+02: +0.01, 1e+03: -0.13, 3e+03: +0.20, 1e+04: +0.07, 3e+04: +0.15, 1e+05: +0.02, 3e+05: +0.08, 1e+06: +0.08
- optimal polynomial intersection (OPI), m = 2002: 1e+02: +0.03, 3e+02: +0.10, 1e+03: +0.09, 3e+03: -0.08, 1e+04: +0.07, 3e+04: +0.06, 1e+05: +0.06, 3e+05: +0.14
- optimal polynomial intersection (OPI), m = 4000: 1e+02: +0.05, 3e+02: -0.01, 1e+03: +0.09, 3e+03: +0.04, 1e+04: -0.04, 3e+04: +0.04, 1e+05: +0.14

## Longer runs on some of the seeds only (not in the table above)

| family | m | search | moves per chain | seeds | mean over chains | best chain |
|---|---|---|---|---|---|---|
| alternant max-agreement (AMA), s = 6 | 14000 | annealing | 1e+06 | 3101 | 0.5193 | 0.5201 |
| alternant max-agreement (AMA), s = 6 | 14000 | line search | 3e+05 | 3101 | 0.5189 | 0.5215 |

## Line search along the run (from 1000 moves on): z minus zhat(N)

| family | m | mean over chains: lowest | highest | best chain: lowest | highest | at (seed, moves) |
|---|---|---|---|---|---|---|
| alternant max-agreement (AMA), s = 6 | 2000 | +0.00 | +0.36 | -0.23 | +1.36 | 3101, 1536 |
| alternant max-agreement (AMA), s = 6 | 6000 | -0.06 | +0.18 | -0.34 | +0.31 | 3101, 12288 |
| alternant max-agreement (AMA), s = 6 | 14000 | -0.11 | +0.32 | -0.34 | +1.30 | 3103, 1536 |
| alternant max-agreement (AMA), s = 6 | 20000 | -0.03 | +0.13 | -0.31 | +0.22 | 3101, 8192 |
| multiplicative polynomial intersection (MPI) | 600 | -0.33 | -0.03 | -0.55 | +0.46 | 3103, 131072 |
| multiplicative polynomial intersection (MPI) | 1000 | -0.30 | -0.13 | -0.50 | -0.10 | 3101, 32768 |
| multiplicative polynomial intersection (MPI) | 2000 | -0.30 | +0.08 | -0.55 | +0.25 | 3103, 49152 |
| multiplicative polynomial intersection (MPI) | 4000 | -0.20 | -0.01 | -0.39 | +0.04 | 3101, 1000000 |
| multiplicative polynomial intersection (MPI) | 6000 | -0.23 | +0.10 | -0.40 | +0.36 | 3102, 3072 |
| optimal polynomial intersection (OPI) | 400 | -0.13 | +0.28 | -0.29 | +0.28 | 2101, 3072 |
| optimal polynomial intersection (OPI) | 1008 | -0.02 | +0.27 | -0.05 | +0.77 | 2101, 8192 |
| optimal polynomial intersection (OPI) | 2002 | -0.06 | +0.18 | -0.24 | +0.40 | 2101, 24576 |
| optimal polynomial intersection (OPI) | 4000 | -0.04 | +0.25 | -0.26 | +0.23 | 2102, 1024 |

## AMA keys at longer lengths (p = 11, s = 6, alpha = 0.08)

| m | r | l | DQI (finite) | z | log2 N | z of DQI - 0.01 | log2 N | log2 field operations |
|---|---|---|---|---|---|---|---|---|
| 14000 | 187 | 93 | 0.5321 | 8.37 | 55 | 5.90 | 29 | 49 |
| 20000 | 267 | 133 | 0.5329 | 10.26 | 81 | 7.30 | 43 | 64 |
| 30000 | 400 | 199 | 0.5335 | 12.81 | 123 | 9.18 | 65 | 88 |
| 40000 | 533 | 266 | 0.5340 | 15.02 | 168 | 10.83 | 89 | 113 |
| 50000 | 667 | 333 | 0.5343 | 16.92 | 212 | 12.24 | 113 | 137 |

## AMA, m = 14000: searches with the key and a search without it, same start, same number of moves

| seed | moves | start | with the key, line search | with the key, annealing | with the key, random start | without the key, line search | DQI (finite) |
|---|---|---|---|---|---|---|---|
| 3101 | 10000 | 0.5144 | 0.5144 | 0.5144 | 0.4861 | 0.5162 | 0.5321 |
| 3102 | 10000 | 0.5151 | 0.5151 | 0.5151 | 0.4859 | 0.5156 | 0.5321 |
| 3103 | 10000 | 0.5171 | 0.5171 | 0.5171 | 0.4841 | 0.5171 | 0.5321 |

## Row-subset lattice attack on MPI: mean over subsets at the best subset size

(*: more than half of the subsets reached the time limit at this block size)

| m | n | seeds | DQI (finite) | block | rows | subsets | mean | standard error | best subset | model on the stored profiles |
|---|---|---|---|---|---|---|---|---|---|---|
| 600 | 42 | 1 | 0.2235 | 2 | 160 | 16 | 0.2030 | 0.0061 | 0.2474 | 0.1905 |
| 600 | 42 | 1 | 0.2235 | 20 | 130 | 16 | 0.2110 | 0.0044 | 0.2402 | 0.2072 |
| 600 | 42 | 1 | 0.2235 | 30 | 260 | 16 | 0.2419 | 0.0058 | 0.2802 | 0.2238 |
| 600 | 42 | 1 | 0.2235 | 40 | 160 | 16 | 0.2400 | 0.0052 | 0.2819 | 0.2360 |
| 600 | 42 | 1 | 0.2235 | 50 | 260 | 16 | 0.2495 | 0.0069 | 0.3020 | 0.2401 |
| 600 | 42 | 1 | 0.2235 | 60 | 260 | 16 | 0.2502 | 0.0052 | 0.2825 | 0.2464 |
| 1000 | 70 | 1 | 0.2338 | 2 | 340 | 16 | 0.1601 | 0.0057 | 0.1983 | 0.1525 |
| 1000 | 70 | 1 | 0.2338 | 20 | 280 | 16 | 0.1731 | 0.0059 | 0.2093 | 0.1645 |
| 1000 | 70 | 1 | 0.2338 | 30 | 180 | 16 | 0.1780 | 0.0049 | 0.2151 | 0.1739 |
| 1000 | 70 | 1 | 0.2338 | 40 | 220 | 16 | 0.1856 | 0.0042 | 0.2270 | 0.1855 |
| 1000 | 70 | 1 | 0.2338 | 50 | 280 | 16 | 0.1959 | 0.0052 | 0.2241 | 0.1929 |
| 1000 | 70 | 1 | 0.2338 | 60 | 340 | 16 | 0.2038 | 0.0049 | 0.2400 | 0.1994 |
| 2000 | 140 | 3 | 0.2433 | 2 | 220 | 12 | 0.1088 | 0.0038 | 0.1349 | 0.1085 |
| 2000 | 140 | 3 | 0.2433 | 20 | 260 | 28 | 0.1152 | 0.0023 | 0.1386 | 0.1168 |
| 2000 | 140 | 3 | 0.2433 | 30 | 350 | 12 | 0.1265 | 0.0026 | 0.1404 | 0.1232 |
| 2000 | 140 | 3 | 0.2433 | 40 | 500 | 12 | 0.1306 | 0.0035 | 0.1547 | 0.1312 |
| 2000 | 140 | 3 | 0.2433 | 50 | 350 | 12 | 0.1412 | 0.0024 | 0.1586 | 0.1388 |
| 2000 | 140 | 3 | 0.2433 | 60* | 400 | 28 | 0.1485 | 0.0022 | 0.1726 | 0.1440 |
| 6000 | 420 | 1 | 0.2518 | 2 | 480 | 8 | 0.0870 | 0.0027 | 0.0997 | 0.0799 |
| 6000 | 420 | 1 | 0.2518 | 20 | 480 | 8 | 0.0832 | 0.0023 | 0.0961 | 0.0800 |
| 6000 | 420 | 1 | 0.2518 | 30 | 560 | 8 | 0.0811 | 0.0029 | 0.0937 | 0.0850 |
| 6000 | 420 | 1 | 0.2518 | 40 | 560 | 8 | 0.0836 | 0.0028 | 0.0980 | 0.0875 |
| 6000 | 420 | 1 | 0.2518 | 50* | 560 | 8 | 0.0864 | 0.0022 | 0.0983 | 0.0896 |

## Lattice model against the measurements, all configurations

| m | configurations | rms difference | mean standard error | largest overestimate by the model | largest underestimate |
|---|---|---|---|---|---|
| 600 | 36 | 0.0066 | 0.0063 | 0.0074 | 0.0180 |
| 1000 | 36 | 0.0056 | 0.0051 | 0.0122 | 0.0086 |
| 2000 | 36 | 0.0156 | 0.0034 | 0.0440 | 0.0063 |
| 6000 | 20 | 0.0214 | 0.0029 | 0.0417 | 0.0071 |

Measured root-Hermite factors (slope of the profiles; at most 8 tours per block size): LLL 1.0222, BKZ-20 1.0186, BKZ-30 1.0161, BKZ-40 1.0143, BKZ-50 1.0128, BKZ-60 1.0119
Estimate used for larger block sizes: BKZ-50 1.0121, BKZ-60 1.0115

The model with the estimated factor against the measured means: m = 600, BKZ-50: 0.247 against 0.249; m = 600, BKZ-60: 0.252 against 0.250; m = 1000, BKZ-50: 0.199 against 0.196; m = 1000, BKZ-60: 0.204 against 0.204; m = 2000, BKZ-50: 0.147 against 0.141; m = 2000, BKZ-60: 0.150 against 0.148; m = 6000, BKZ-50: 0.096 against 0.086

## Lattice model: block size at which the modeled attack reaches DQI's value (M = 8191, alpha = 0.07)

| m | n | DQI (finite) | block size | vectors on the slope | 0.292 x block size |
|---|---|---|---|---|---|
| 600 | 42 | 0.2235 | at most 50 | - | - |
| 800 | 56 | 0.2297 | 71 | 207 | 21 |
| 1000 | 70 | 0.2338 | 125 | 265 | 37 |
| 1500 | 105 | 0.2411 | 278 | 410 | 81 |
| 2000 | 140 | 0.2433 | 439 | 555 | 128 |
| 3000 | 210 | 0.2471 | 812 | 845 | 237 |
| 4000 | 280 | 0.2493 | not reached while the block is shorter than the slope (model: 0.2410 at block size 1095) | - | above 320 |
| 6000 | 420 | 0.2518 | not reached while the block is shorter than the slope (model: 0.2270 at block size 1535) | - | above 448 |
