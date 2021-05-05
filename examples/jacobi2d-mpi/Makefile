TARGET = jacobi2d

NVCC_OPTS = -arch=sm_70 -I$(MPI_ROOT)/include
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart
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
