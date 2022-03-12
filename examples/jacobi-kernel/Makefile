NVCC = nvcc
ARCH = -arch=sm_70
NVCC_CU_OPTS = --std=c++11 $(ARCH)
NVCC_LINK_OPTS = $(ARCH) -lcuda

TARGET = jacobi_kernel

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(NVCC) $(NVCC_LINK_OPTS) -o $@ $^

%.o: %.cu %.h
	$(NVCC) $(NVCC_CU_OPTS) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
