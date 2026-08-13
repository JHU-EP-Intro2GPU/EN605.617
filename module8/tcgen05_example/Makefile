NVCC      := nvcc
ARCH      := sm_100a
NVCC_FLAGS:= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SRC := src/tcgen05_gemm.cu
DEP := src/common.cuh
BIN := tcgen05_gemm

all: $(BIN)

$(BIN): $(SRC) $(DEP)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN) $(SRC)

run: all
	./$(BIN)

ptx: $(SRC) $(DEP)
	$(NVCC) $(NVCC_FLAGS) -ptx -o tcgen05_gemm.ptx $(SRC)

clean:
	rm -f $(BIN) tcgen05_gemm.ptx

.PHONY: all run ptx clean
