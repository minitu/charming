TARGET = baseline

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).cu
	nvcc -arch=sm_70 -o $@ $^

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
