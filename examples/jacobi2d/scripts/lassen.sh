#!/bin/bash
#BSUB -G asccasc
#BSUB -q pbatch
#BSUB -W 10
#BSUB -nnodes 1
#BSUB -J jacobi2d-strong-n1
#BSUB -o jacobi2d-strong-n1.%J
#BSUB -alloc_flags "smt4"

n_nodes=1
block_width=16384
block_height=16384
n_iters=10

n_procs=$(($n_nodes * 4))

set -x
cd $HOME/charming/examples/jacobi2d

date

jsrun -n$n_procs -a1 -c1 -g1 -r4 ./jacobi2d-bench $block_width $block_height $n_procs $n_iters

date
