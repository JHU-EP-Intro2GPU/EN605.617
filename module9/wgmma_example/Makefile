NVCC      := nvcc
ARCH      := sm_90a
NVCC_FLAGS:= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SRC := src/wgmma_gemm.cu
DEP := src/common.cuh
BIN := wgmma_gemm

all: $(BIN)

$(BIN): $(SRC) $(DEP)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN) $(SRC)

run: all
	./$(BIN)

ptx: $(SRC) $(DEP)
	$(NVCC) $(NVCC_FLAGS) -ptx -o wgmma_gemm.ptx $(SRC)

clean:
	rm -f $(BIN) wgmma_gemm.ptx

.PHONY: all run ptx clean
