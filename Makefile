TARGET = nvcharm
HEADERS = Message.h

$(TARGET): $(TARGET).cu $(HEADERS)
	nvcc --std=c++11 -o $@ $<

clean:
	rm -f $(TARGET)
