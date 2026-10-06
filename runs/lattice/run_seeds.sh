#!/bin/bash
# Lattice attack, seeds 3102 and 3103 at m=600 and m=2000.
cd "$(dirname "$0")/../.." || exit 1
L=results/lattice
P=${FPYLLL_PYTHON:-python}
S=scripts/43_lattice_attack.py
$P $S $L/logcauchy-m600-s3102.json --subset-rows 100,150 --blocks 20 --seconds 300 > $L/run-m600-s3102.log 2>&1 &
$P $S $L/logcauchy-m600-s3103.json --subset-rows 100,150 --blocks 20 --seconds 300 > $L/run-m600-s3103.log 2>&1 &
$P $S $L/logcauchy-m2000-s3102.json --subset-rows 180,220 --blocks 2,20 --seconds 1200 > $L/run-m2000-s3102.log 2>&1 &
$P $S $L/logcauchy-m2000-s3103.json --subset-rows 180,220 --blocks 2,20 --seconds 1200 > $L/run-m2000-s3103.log 2>&1 &
wait
echo ALLDONE
tail -n 6 $L/run-m600-s310*.log $L/run-m2000-s310*.log
