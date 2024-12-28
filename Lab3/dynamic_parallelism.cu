/*
CUDA - dynamic parallelism sample
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}


__device__ void swap(int* a, int* b){
	int temp = *a;
	*a = *b;
	*b = temp;
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);  
    return i + 1;
}


__global__ 
void quickSort(int* d_array, int low, int high) {
	if (low < high) {
        int pivot = partition(d_array, low, high);

        quickSort<<<1,1>>>(d_array, low, pivot - 1);
        quickSort<<<1,1>>>(d_array, pivot + 1, high);
    }
}

void generateRandomNumbers(int *arr, int n) {
	srand(time(NULL));

    for (int i = 0; i < n; i++) {
        arr[i] = -10000 + rand() % (10000 - (-10000) + 1);
	}

}

void printArray(int* arr, int n){
	for(int i=0;i<n;i++){
		printf("%d: %d\n", i, arr[i]);
	}
}

int main(int argc,char **argv) {
	float milliseconds;
	const int N = 100000;
	int* h_array = (int* )malloc(sizeof(int) * N);
	generateRandomNumbers(h_array, N);

	int* d_array;

	printf("-------------INITIAL-------------\n");
	printArray(h_array, N);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&d_array, N * sizeof(int));
	cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

	//run kernel on GPU 
	quickSort<<<1, 1>>>(d_array, 0, N-1);

	cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

	printf("-------------SORTED-------------\n");
	printArray(h_array, N);

	printf("Kernel execution time: %.3f ms\n", milliseconds);

	free(h_array);
    cudaFree(d_array);
}
