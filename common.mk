# Whether to build with MPI support
CHARMING_USE_MPI ?= 0
# Communication mechanism - 0: Get-based (default), 1: Put-based MPSC, 2: Put-based SPSC
CHARMING_COMM_TYPE ?= 0

NVSHMEM_PREFIX ?= $(HOME)/nvshmem/install
CHARMING_PREFIX ?= $(HOME)/work/charming/install

NVCC = nvcc
ARCH = -arch=sm_70
OPTS = #-DNO_CLEANUP -DDEBUG
