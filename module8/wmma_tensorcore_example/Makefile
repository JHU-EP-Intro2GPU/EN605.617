NVCC      := nvcc
ARCH      := sm_80
NVCC_FLAGS:= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SRC := src/wmma_gemm.cu
BIN := wmma_gemm

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN) $(SRC)

run: all
	./$(BIN)

ptx: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -ptx -o wmma_gemm.ptx $(SRC)

clean:
	rm -f $(BIN) wmma_gemm.ptx

.PHONY: all run ptx clean
