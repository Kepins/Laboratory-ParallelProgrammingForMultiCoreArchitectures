#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeSum(int *d_input, int N, long long *d_result) {
    __shared__ long long sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    long long my_value = (i < N) ? d_input[i] : 0;
    sdata[tid] = my_value;
    __syncthreads ();

    // Perform block-wide reduction using shared memory

    //V1
    // for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // V2 not working
    // for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // if ( tid < 32) {
    //     sdata [ tid ] += sdata [tid + 32];
    //     sdata [ tid ] += sdata [tid + 16];
    //     sdata [ tid ] += sdata [tid + 8];
    //     sdata [ tid ] += sdata [tid + 4];
    //     sdata [ tid ] += sdata [tid + 2];
    //     sdata [ tid ] += sdata [tid + 1];
    // }

    // V3
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if ( tid < 32) {
        sdata [ tid ] += sdata [tid + 32];
        __syncwarp();
        sdata [ tid ] += sdata [tid + 16];
        __syncwarp();
        sdata [ tid ] += sdata [tid + 8];
        __syncwarp();
        sdata [ tid ] += sdata [tid + 4];
        __syncwarp();
        sdata [ tid ] += sdata [tid + 2];
        __syncwarp();
        sdata [ tid ] += sdata [tid + 1];
        __syncwarp();
    }


    if (tid == 0){
        atomicAdd((unsigned long long *)&d_result[0], sdata[0]);
    }
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

    // Initialize d_results to 0
    cudaMemset(d_results, 0, 1 * sizeof(long long));

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
