# module0 Makefile (minimal)

.PHONY: clean

hello-world-cuda: hello-world.cu
	nvcc -O2 hello-world.cu -o hello-world-cuda

hello-world-opencl: hello_world_cl.c
	gcc -O2 hello_world_cl.c -o hello-world-opencl -lOpenCL

clean:
	rm -f hello-world-cuda hello-world-opencl