all:
	nvcc assignment.cu -o assignment.exe -std=c++11
	nvcc assignment_advanced.cu -o assignment_advanced.exe -std=c++11

clean:
	rm assignment.exe
	rm assignment_advanced.exe

test:
	./run.sh 64 64
