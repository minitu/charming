TARGET = charming

BUILD_DIR = build
NVSHMEM_HOME ?= $(HOME)/nvshmem/install
NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(MPI_ROOT)/include -I./include #-DDEBUG
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

HEADERS = include/$(TARGET).h \
          include/message.h \
          include/scheduler.h \
          include/chare.h \
          src/ringbuf.h \
          src/heap.h \
          src/composite.h \
          src/util.h

OBJS = $(BUILD_DIR)/$(TARGET).o \
       $(BUILD_DIR)/scheduler.o \
       $(BUILD_DIR)/ringbuf.o \
       $(BUILD_DIR)/heap.o \
       $(BUILD_DIR)/util.o

.PHONY: all
all: $(BUILD_DIR) $(BUILD_DIR)/lib$(TARGET).a

$(BUILD_DIR):
	mkdir $@

$(BUILD_DIR)/lib$(TARGET).a: $(OBJS)
	ar cru $@ $^
	ranlib $@

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS)
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -rf build
