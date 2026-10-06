#!/bin/bash
# one m=6000 seed, BKZ-20 on larger subsets; usage: run_m6000_one.sh <seed>
cd "$(dirname "$0")/../.." || exit 1
L=results/lattice
s=$1
[ -f $L/logcauchy-m6000-s$s-bkz.json ] || cp $L/logcauchy-m6000-s$s.json $L/logcauchy-m6000-s$s-bkz.json
${FPYLLL_PYTHON:-python} -u scripts/43_lattice_attack.py $L/logcauchy-m6000-s$s-bkz.json --subset-rows 520,700 --blocks 20 --seconds 2400 --seed $s > $L/run-m6000-s$s-bkz.log 2>&1
echo "exit $?" >> $L/run-m6000-s$s-bkz.log
