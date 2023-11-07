#!/bin/bash
#SBATCH -J ethylene-freq
#SBATCH -o ethylene_freq.%j.%N.out
#SBATCH -p shared
#SBATCH -A itm101
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

module load cpu/0.15.4 gcc/10.2.0 mvapich2/2.3.6
module load qchem/6.0.2

qchem ethylene_frequencies.qcin ethylene_freq.out
