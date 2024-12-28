#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeSum(int *d_input, int N, long long *d_result) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    long my_value = 0;
    if (i < N){
        my_value = long(d_input[i]);
    }
    atomicAdd((unsigned long long *)&d_result[0], my_value);
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
	srand(time(NULL));

    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

int main(int argc,char **argv) {

    int threadsinblock=1024;
    int blocksingrid;

    int N,A,B;
    
 	cudaEvent_t start, stop;
    float milliseconds = 0;
    
    printf("Enter number of elements: \n");
    scanf("%d", &N);


	printf("Enter A value (start range): \n");
    scanf("%d", &A);

    printf("Enter B value (end range): \n");
    scanf("%d", &B);

	int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	generateRandomNumbers(randomNumbers, N,A,B);

	blocksingrid = ceil((double)N/threadsinblock);

	printf("The kernel will run with: %d blocks\n", blocksingrid);

	int *d_input;
    long long * h_results, *d_results;

	h_results = (long long*)calloc(1, sizeof(long long));

	if (h_results == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_results, 1 * sizeof(long long));

    cudaMemcpy(d_input, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device histogram to 0
    // cudaMemset(d_results, 0, (B-A) * sizeof(int));

    computeSum<<<blocksingrid, threadsinblock>>>(d_input, N, d_results);

    // Copy the result back to the host
    cudaMemcpy(h_results, d_results, 1 * sizeof(long long), cudaMemcpyDeviceToHost);
    
    long long sum = h_results[0];

    printf("\nSum: %lld\n\n", sum);
    double avg = double(sum) / N;
    printf("Avg: %lf\n\n", avg);


    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print execution time

	printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Free allocated memory
    free(randomNumbers);
    free(h_results);
    cudaFree(d_input);
    cudaFree(d_results);

    return 0;

}
