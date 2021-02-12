TARGET = nvcharm
HEADERS = $(TARGET).h Message.h
NVCC_OPTS = -arch=sm_70
NVCC_LINK = nvcc $(NVCC_OPTS)
NVCC = nvcc --std=c++11 -dc $(NVCC_OPTS)

$(TARGET): user.o $(TARGET).o
	$(NVCC_LINK) -o $@ $^

$(TARGET).o: $(TARGET).cu $(HEADERS)
	$(NVCC) -o $@ -c $<

user.o: user.cu
	$(NVCC) -o $@ -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
