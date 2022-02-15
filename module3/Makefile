all:
	nvcc assignment.cu -o assignment.exe
	nvcc assignment_advanced.cu -o assignment_advanced.exe

clean:
	rm assignment.exe
	rm assignment_advanced.exe

test:
	./run.sh 64 64
