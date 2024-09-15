# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: cutensor_example.cu
	$(CXX) cutensor_example.cu --std c++17 `pkg-config opencv --cflags --libs` -o cutensor_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcutensor

run:
	./cutensor_example.exe $(ARGS)

clean:
	rm -f cutensor_example.exe output*.txt 