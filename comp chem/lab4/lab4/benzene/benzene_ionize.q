#!/bin/bash
#SBATCH -J benzene-ip
#SBATCH -o benzene_ip.%j.%N.out
#SBATCH -p shared
#SBATCH -A itm101
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00

module load cpu/0.15.4 gcc/10.2.0 mvapich2/2.3.6
module load qchem/6.0.2

qchem benzene_ionize.qcin benzene_ionize.out
