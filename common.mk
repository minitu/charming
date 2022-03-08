# Whether to build with MPI support
CHARMING_USE_MPI ?= 0

NVSHMEM_HOME ?= $(HOME)/nvshmem/install
CHARMING_HOME ?= $(HOME)/work/charming

NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(CHARMING_HOME)/include
ifeq ($(CHARMING_USE_MPI), 1)
NVCC_OPTS += -I$(MPI_ROOT)/include -DCHARMING_USE_MPI
endif
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

ifeq ($(CHARMING_USE_MPI), 1)
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart -L$(CHARMING_HOME)/build -lcharming
else
NVCC_LINK = nvcc $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -lcudart -L$(CHARMING_HOME)/build -lcharming
endif
