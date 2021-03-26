TARGET = nvcharm
NVSHMEM_HOME ?= $(HOME)/nvshmem/install
NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(MPI_ROOT)/include -DDEBUG
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

HEADERS = $(TARGET).h message.h scheduler.h chare.h ringbuf.h util.h user.h
OBJS = $(TARGET).o scheduler.o ringbuf.o util.o user.o

$(TARGET): $(OBJS)
	$(NVCC_LINK) -o $@ $^

$(TARGET).o: $(TARGET).cu $(HEADERS)
	$(NVCC) -o $@ -c $<

%.o: %.cu %.h
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
