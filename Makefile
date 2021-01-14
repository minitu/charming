TARGET = nvcharm

$(TARGET): $(TARGET).cu
	nvcc --std=c++11 -o $@ $<

clean:
	rm -f $(TARGET)
