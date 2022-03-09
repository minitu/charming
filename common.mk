# Whether to build with MPI support
CHARMING_USE_MPI ?= 0

NVSHMEM_PREFIX ?= $(HOME)/nvshmem/install
CHARMING_PREFIX ?= $(HOME)/work/charming/install

NVCC = nvcc
ARCH = -arch=sm_70
