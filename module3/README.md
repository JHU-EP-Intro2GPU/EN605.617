Instructions:

Create a program that contains the same (or similar) algorithm executing using a CPU-based method and a CUDA kernel.  There should be minimal conditional branching in the method and kernel.  This should use a non-trivial amount of data 1000s to millions, however much you would like to test.
Create a separate program (or include in the same code base from the previous step) that demonstrates the effect of conditional branching on implementing the same CPU and GPU code.  You will also need to include at least one performance comparison chart and a short text file that includes your thoughts on the results.
The base CUDA source code file must be called assignment.cu and be housed in the module3 directory of your repository. It will need to be runnable as assignment.exe 512 256.  You can modify the Makefile as you see fit, as long as the make command (with no arguments), builds the assignment.exe as the executable output.  The assignment.cu file that I have provided in the module3 directory handles the two arguments for total number of threads and number of threads per block. Note when running the CPU algorithm you will not need to take these arguments into account as you don't have a lot of control over that.
You will need to include the zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.
 Look at the below example of a main function for the same assignment in a previous year.  If the surrounding script only executes the main function once, what is good and bad about this submission? Do not fix the assignment or submit it along with your other code.  Just give me a short description of your thoughts about it.
 

int main(int argc,char* argv[]) {

outputCardInfo();
int blocks = 3;
int threads = 64;
if (argc == 2) {

blocks = atoi(argv[1]);
printf("Blocks changed to:%i\n", blocks);

} else if (argc == 3) {

blocks = atoi(argv[1]);
threads = atoi(argv[2]);
printf("Blocks changed to:%i\n", blocks);
printf("Threads changed to:%i\n", threads);

}

int a[N], b[N], c[N];
int *dev_a, *dev_b, *dev_c;
cudaMalloc((void**)&dev_a, N * sizeof(int));
cudaMalloc((void**)&dev_b, N * sizeof(int));
cudaMalloc((void**)&dev_c, N * sizeof(int));

//Populate our arrays with numbers.
for (int i = 0; i < N; i++) {

a[i] = -i;
b[i] = i*i;

}

cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
auto start = std::chrono::high_resolution_clock::now();
add<<<blocks,threads>>> (dev_a, dev_b, dev_c);
auto stop = std::chrono::high_resolution_clock::now();
cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
auto startHost = std::chrono::high_resolution_clock::now();
addHost(a, b, c);
auto stopHost = std::chrono::high_resolution_clock::now();
std::cout <endl<< " Time elapsed GPU = " << std::chrono::duration_castchrono::nanoseconds>(stop - start).count() << "ns\n";
std::cout << " Time elapsed Host = " << std::chrono::duration_castchrono::nanoseconds>(stopHost - startHost).count() << "ns\n";
return 0;

}