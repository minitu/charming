TARGET = nvcharm
HEADERS = Message.h
NVCC_LINK = nvcc
NVCC = nvcc --std=c++11 -dc

$(TARGET): user.o $(TARGET).o
	$(NVCC_LINK) -o $@ $^

$(TARGET).o: $(TARGET).cu $(HEADERS)
	$(NVCC) -o $@ -c $<

user.o: user.cu
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
