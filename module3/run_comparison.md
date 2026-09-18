# Module 3 Assignment — Run Comparison

| Case | Total Threads | Threads per Block | CPU (no branch) µs | CUDA (no branch) µs | CPU (branch) µs | CUDA (branch) µs |
|-----:|--------------:|------------------:|-------------------:|--------------------:|----------------:|-----------------:|
| 1    | 512           | 256               | 9937.904           | 7010.303           | 10648.919       | 6970.272         |
| 2    | 4 096         | 256               | 9788.010           | 1043.743           | 10318.783       | 1158.720         |
| 3    | 1 048 576     | 256               | 9516.269           | 453.631            | 15603.582       | 872.928          |
| 4    | 4 096         | 512               | 10155.794          | 1357.887           | 11470.969       | 1205.088         |
| 5    | 1 048 576     | 512               | 10099.789          | 441.536            | 10842.741       | 551.199          |
| 6    | 4 224 (was 4096) | 192            | 9883.403           | 1012.959           | 10705.082       | 1701.184         |
| 7    | 1 048 704 (was 1048576) | 192     | 10413.070          | 448.511            | 10803.374       | 546.624          |

## Notes:
- I noticed that the CUDA non-branching code appeared to take a lot longer to run than the CUDA branching code and learned that the startup overhead of the first kernel was the cause. To make a more fair comparison of the two kernels, I run a warm-up kernel first before timing any code.
- I wasn't sure if it's best practice to include the overhead of the CUDA-specific memory operations (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) in the timing of the CUDA code. I ended up not including it because I wanted to compare the performance of just the main processing loops, but I can definitely see how the additional overhead may sway the overall time taken in favor of the CPU method.

## Thoughts:
- No real surprises for the CPU results themselves: generally very consistent, with the branching code taking a little longer.
- For the CUDA kernels, it does appear that throwing more threads at the problem made it run faster (I used 10 million elements in my arrays so this makes sense).
- The branching CUDA kernel appears to take no more than twice the amount of time as the non-branching kernel, and in some cases its running time is almost comparable, which is interesting. 