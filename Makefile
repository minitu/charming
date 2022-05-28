include common.mk

TARGET := charming
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
INC_DIR := $(BUILD_DIR)/include
LIB_DIR := $(BUILD_DIR)/lib

SRCS = $(TARGET).cu \
       sched/scheduler.cu \
       util/util.cu

ifeq ($(CHARMING_COMM_TYPE), 0)
SRCS += comm/get/ringbuf.cu \
        comm/get/heap.cu \
        comm/get/get.cu
else
SRCS += comm/put/ringbuf.cu
ifeq ($(CHARMING_COMM_TYPE), 1)
SRCS += comm/put/put_mpsc.cu
else
ifeq ($(CHARMING_COMM_TYPE), 2)
SRCS += comm/put/msg_queue.cu \
        comm/put/put_spsc.cu
endif
endif
endif

OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(filter %.cu, $(SRCS)))

INC = -Iinclude -Isrc/comm -Isrc/sched -Isrc/util
ifeq ($(CHARMING_COMM_TYPE), 0)
INC += -Isrc/comm/get
else
INC += -Isrc/comm/put
endif

NVCC_CU_OPTS = --std=c++11 -dc $(ARCH) $(OPTS) -DCHARMING_COMM_TYPE=$(CHARMING_COMM_TYPE) -I$(NVSHMEM_PREFIX)/include
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
