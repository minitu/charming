#!/bin/bash
#BSUB -W 0:10
#BSUB -nnodes 128
#BSUB -P csc357
#BSUB -J jacobi1d-strong-n128
#BSUB -o jacobi1d-strong-n128.%J
#BSUB -alloc_flags "smt1"

n_nodes=128
block_width=1000000
n_iters=1000

n_procs=$(($n_nodes * 6))

set -x
cd $HOME/work/charming/examples/jacobi1d
export LD_LIBRARY_PATH=/sw/summit/gdrcopy/2.0/lib64:$LD_LIBRARY_PATH

date

jsrun -n$n_procs -a1 -c1 -g1 -r6 ./jacobi1d-bench $block_width $n_iters

date
