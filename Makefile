include common.mk

TARGET = charming
BUILD_DIR = build

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
