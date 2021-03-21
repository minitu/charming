TARGET = nvcharm
HEADERS = $(TARGET).h message.h
NVSHMEM_HOME ?= $(HOME)/nvshmem_src_2.0.3-0/install
NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(MPI_ROOT)/include
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

$(TARGET): $(TARGET).o user.o ringbuf.o
	$(NVCC_LINK) -o $@ $^

$(TARGET).o: $(TARGET).cu $(HEADERS)
	$(NVCC) -o $@ -c $<

user.o: user.cu user.h
	$(NVCC) -o $@ -c $<

ringbuf.o: ringbuf.cu ringbuf.h
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
