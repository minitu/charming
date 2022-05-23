include ../../common.mk
include ../examples.mk

NVCC_CU_OPTS += #-DDEBUG

TARGET = refnum

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(NVCC) $(NVCC_LINK_OPTS) -o $@ $^

%.o: %.cu %.h
	$(NVCC) $(NVCC_CU_OPTS) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
