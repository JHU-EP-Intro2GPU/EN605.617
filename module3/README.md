This project implements CPU and GPU versions of array arithmetic with conditional branching.

### Build

Run:

```bash
./build.sh
```

This will generate:

* `assignment.exe` — GPU version
* `assignment_cpu.exe` — CPU version

### Run

To run the GPU version:

```bash
./run.sh <numThreads> <threadsPerBlock>
```

or:

```bash
./assignment.exe <numThreads> <threadsPerBlock>
```

For example:

```bash
./assignment.exe 512 256
```

To run the CPU version:

```bash
./assignment_cpu.exe <numElements> <threadsPerBlock>
```

The `threadsPerBlock` argument is ignored by the CPU version because the CPU implementation does not use CUDA threads or blocks.

If the requested number of threads is not evenly divisible by the block size, the GPU program rounds the total number of threads up to the next complete block.
