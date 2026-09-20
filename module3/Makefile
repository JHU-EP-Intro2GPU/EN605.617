all: assignment.exe assignment_cpu.exe

assignment.exe: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe

assignment_cpu.exe: assignment.c
	gcc assignment.c -o assignment_cpu.exe