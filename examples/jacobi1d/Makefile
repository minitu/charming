include ../../common.mk

TARGET = jacobi1d

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(NVCC_LINK) -o $@ $^

%.o: %.cu %.h
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
