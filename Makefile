TARGET = charming

BUILD_DIR = build
NVSHMEM_HOME ?= $(HOME)/nvshmem/install
NVCC_OPTS = -arch=sm_70 -I$(NVSHMEM_HOME)/include -I$(MPI_ROOT)/include -I./include -DDEBUG
NVCC_LINK = nvcc -ccbin=mpicxx $(NVCC_OPTS) -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_ROOT)/lib -lmpi_ibm -lcuda -lcudart
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

HEADERS = include/$(TARGET).h \
          src/message.h \
          src/scheduler.h \
          src/chare.h \
          src/ringbuf.h \
          src/util.h

OBJS = $(BUILD_DIR)/$(TARGET).o \
       $(BUILD_DIR)/scheduler.o \
       $(BUILD_DIR)/ringbuf.o \
       $(BUILD_DIR)/util.o

.PHONY: all
all: $(BUILD_DIR) $(BUILD_DIR)/lib$(TARGET).a

$(BUILD_DIR):
	mkdir $@

$(BUILD_DIR)/lib$(TARGET).a: $(OBJS)
	ar cru $@ $^
	ranlib $@

$(BUILD_DIR)/$(TARGET).o: src/$(TARGET).cu $(HEADERS)
	$(NVCC) -o $@ -c $<

$(BUILD_DIR)/%.o: src/%.cu src/%.h
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -rf build
