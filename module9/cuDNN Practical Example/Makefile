# IDIR=./
CXX = g++

all: clean build

build: cudnn_example.cpp
	$(CXX) cudnn_example.cpp --std c++17 -o cudnn_example.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -lcuda -lcudnn

run:
	./cudnn_example.exe $(ARGS)

clean:
	rm -f cudnn_example.exe output*.txt 