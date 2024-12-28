/*
CUDA - prepare the histogram of N numbers in range of <a;b> where a and b should be integers
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeHistogram(int *d_input, int *d_output, int N, int A, int B, int streamChunk, int streamId) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x+streamId*streamChunk;
    if (idx < N  && idx < (streamId + 1) * streamChunk) {
        // Calculate the bin index for the data value
        int resultIdx = (d_input[idx] - A);
        atomicAdd(&d_output[resultIdx], 1); // Atomic update to avoid race conditions
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
	srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A);
    }
}

int main(int argc,char **argv) {
    ///define number of streams
    int numberOfStreams = 4;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    //create streams
    cudaStream_t streams[numberOfStreams];
    for(int i=0;i<numberOfStreams;i++) {
        if (cudaSuccess!=cudaStreamCreate(&streams[i])) {
            errorexit("Error creating stream");
        }
    }

    int N,A,B;
    
    printf("Enter number of elements: \n");
    scanf("%d", &N);

	printf("Enter A value (start range): \n");
    scanf("%d", &A);

    printf("Enter B value (end range): \n");
    scanf("%d", &B);

	//int *randomNumbers = (int *)malloc(N * sizeof(int));
    // if (randomNumbers == NULL) {
    //     printf("Memory allocation failed.\n");
    //     return 1;
    // }
    int * randomNumbers;
    cudaHostAlloc(&randomNumbers, N * sizeof(int), cudaHostAllocDefault);
    

	generateRandomNumbers(randomNumbers, N,A,B);

    //get number of chunks to operate per stream
    int streamChunk = 1 + ((N - 1)/numberOfStreams);

    printf("Stream chunk is %d \n", streamChunk);
    //define kernel size per stream
    int threadsinblock=1024;
    int blocksingrid=1+((streamChunk-1)/threadsinblock); 
    printf("The kernel will run with: %d blocks\n", blocksingrid);

	int *h_result, *d_result, *d_input;

    // (int *)calloc((B-A), sizeof(int));
	cudaHostAlloc(&h_result, (B-A) * sizeof(int), cudaHostAllocDefault);

	if (h_result == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_result, (B-A) * sizeof(int));

    // Initialize device histogram to 0
    cudaMemset(d_result, 0, (B-A) * sizeof(int));

    //computeHistogram<<<blocksingrid, threadsinblock>>>(d_input, d_result, N, A, B);

    for(int i=0; i<numberOfStreams; i++) {
        cudaMemcpyAsync(&d_input[streamChunk*i], &randomNumbers[streamChunk*i], streamChunk*sizeof(int), cudaMemcpyHostToDevice, streams[i]);   
        computeHistogram<<<blocksingrid, threadsinblock, threadsinblock*sizeof(int), streams[i]>>>(d_input, d_result, N, A, B, streamChunk, i);
    }
    cudaDeviceSynchronize();

    // Copy the histogram result back to the host
    cudaMemcpy(h_result, d_result, (B-A) * sizeof(int), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print the histogram
    printf("Histogram:\n");
    for (int i = 0; i < B-A; i++) {
        printf("%d occures %d\n", i, h_result[i]);
    }

    // Print execution time
	printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Free allocated memory
    cudaFreeHost(randomNumbers);
    cudaFreeHost(h_result);
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}