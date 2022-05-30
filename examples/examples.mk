NVCC_CU_OPTS = --std=c++11 -dc $(ARCH) $(OPTS) -I$(NVSHMEM_PREFIX)/include -I$(CHARMING_PREFIX)/include

ifeq ($(CHARMING_USE_MPI), 1)
NVCC_LINK_OPTS = -ccbin=mpicxx $(ARCH) -L$(NVSHMEM_PREFIX)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart -L$(CHARMING_PREFIX)/lib -lcharming
else
NVCC_LINK_OPTS = $(ARCH) -L$(NVSHMEM_PREFIX)/lib -lnvshmem -lcuda -lcudart -L$(CHARMING_PREFIX)/lib -lcharming
endif
