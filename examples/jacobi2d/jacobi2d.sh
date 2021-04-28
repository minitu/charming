#!/bin/bash
#BSUB -W 0:10
#BSUB -nnodes 128
#BSUB -P csc357
#BSUB -J jacobi2d-strong-n128
#BSUB -o jacobi2d-strong-n128.%J
#BSUB -alloc_flags "smt1"

n_nodes=128
block_width=16384
block_height=16384
n_iters=100

n_procs=$(($n_nodes * 6))

set -x
cd $HOME/work/charming/examples/jacobi2d
export LD_LIBRARY_PATH=/sw/summit/gdrcopy/2.0/lib64:$LD_LIBRARY_PATH

date

jsrun -n$n_procs -a1 -c1 -g1 -r6 ./jacobi2d-bench $block_width $block_height $n_iters

date
