all: build run postprocess

build: assignment.cu
	nvcc --std=c++17 --extended-lambda \
		assignment.cu -o assignment.exe -lcurand

run: assignment.exe
	./assignment.exe

postprocess:
	uv run postprocess.py

clean:
	rm -f assignment.exe
	rm -rf data/ charts/

clean-all: clean
	rm -rf .venv __pycache__
