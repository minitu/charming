TARGET = jacobi1d

CHARMING_HOME = ../..
NVSHMEM_HOME ?= $(HOME)/nvshmem/install
NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(CHARMING_HOME)/include
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart -L$(CHARMING_HOME)/build -lcharming
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(NVCC_LINK) -o $@ $^

%.o: %.cu %.h
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
