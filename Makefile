include common.mk

TARGET := charming
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
INC_DIR := $(BUILD_DIR)/include
LIB_DIR := $(BUILD_DIR)/lib

SRCS := $(TARGET).cu \
        sched/scheduler.cu \
        comm/get/ringbuf.cu \
        comm/get/heap.cu \
        util/util.cu

OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(filter %.cu, $(SRCS)))

INC := -Iinclude -Isrc/comm/get -Isrc/sched -Isrc/util

NVCC_CU_OPTS = --std=c++11 -dc $(ARCH) -I$(NVSHMEM_PREFIX)/include
ifeq ($(CHARMING_USE_MPI), 1)
NVCC_CU_OPTS += -I$(MPI_ROOT)/include -DCHARMING_USE_MPI
endif

.PHONY: default
default: lib

.PHONY: lib
lib: $(LIB_DIR)/lib$(TARGET).a

$(LIB_DIR)/lib$(TARGET).a: $(OBJS)
	@mkdir -p `dirname $@`
	$(NVCC) -lib -o $@ $^

$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p `dirname $@`
	$(NVCC) $(NVCC_CU_OPTS) $(INC) -o $@ -c $<

.PHONY: install
install: lib
	mkdir -p $(CHARMING_PREFIX)/lib
	mkdir -p $(CHARMING_PREFIX)/include
	cp -v $(LIB_DIR)/* $(CHARMING_PREFIX)/lib/
	cp -P -v ./include/* $(CHARMING_PREFIX)/include/

.PHONY: uninstall
uninstall:
	rm -rf $(CHARMING_PREFIX)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: purge
purge: clean uninstall
