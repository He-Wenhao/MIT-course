#!/bin/bash
#SBATCH -J ethylene-ip
#SBATCH -o ethylene_ip.%j.%N.out
#SBATCH -p shared
#SBATCH -A itm101
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

module load cpu/0.15.4  gcc/9.2.0
module load qchem/5.3

qchem ethylene_ionize.qcin ethylene_ionize.out
