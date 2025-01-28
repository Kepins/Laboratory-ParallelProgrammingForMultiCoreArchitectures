#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 2000000000
#define THREADS_PER_BLOCK 256
#define MAX_BLOCK_SIZE 800000
#define CHUNK_SIZE ((int64_t)THREADS_PER_BLOCK * MAX_BLOCK_SIZE)

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in file '%s' at line %d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

__global__ void collatz_iterations(const int64_t *input, int64_t *output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64_t n = input[idx];
        int64_t steps = 0;

        while (n != 1) {
            if (n % 2 == 0)
                n /= 2;
            else
                n = 3 * n + 1;
            steps++;
        }
        output[idx] = steps;
    }
}

int main() {
    // Host memory allocation
    int64_t *h_vector = (int64_t *)malloc(VECTOR_SIZE * sizeof(int64_t));
    int64_t *h_output = (int64_t *)malloc(VECTOR_SIZE * sizeof(int64_t));

    if (!h_vector || !h_output) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Fill the input vector
    #pragma omp parallel
    for (int64_t i = 0; i < VECTOR_SIZE; i++) {
        h_vector[i] = (80000000 + i);
    }

    // Device memory allocation for one chunk
    int64_t *d_vector1, *d_vector2;
    int64_t *d_output1, *d_output2;
    CUDA_CHECK(cudaMalloc((void **)&d_vector1, CHUNK_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_vector2, CHUNK_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_output1, CHUNK_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_output2, CHUNK_SIZE * sizeof(int64_t)));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    double start = omp_get_wtime();

    int64_t chunks = (VECTOR_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE; // Number of chunks

    for (int64_t i = 0; i < chunks; i++) {
        int64_t offset = i * CHUNK_SIZE;
        int64_t current_chunk_size = (VECTOR_SIZE - offset) < CHUNK_SIZE ? (VECTOR_SIZE - offset) : CHUNK_SIZE;

        int64_t blocks = ((current_chunk_size - 1) / THREADS_PER_BLOCK) + 1;

        if (i % 2 == 0) {
            // Process with stream1
            CUDA_CHECK(cudaMemcpyAsync(d_vector1, h_vector + offset, current_chunk_size * sizeof(int64_t), cudaMemcpyHostToDevice, stream1));
            collatz_iterations<<<blocks, THREADS_PER_BLOCK, 0, stream1>>>(d_vector1, d_output1, current_chunk_size);
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
            CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output1, current_chunk_size * sizeof(int64_t), cudaMemcpyDeviceToHost, stream1));
        } else {
            // Process with stream2
            CUDA_CHECK(cudaMemcpyAsync(d_vector2, h_vector + offset, current_chunk_size * sizeof(int64_t), cudaMemcpyHostToDevice, stream2));
            collatz_iterations<<<blocks, THREADS_PER_BLOCK, 0, stream2>>>(d_vector2, d_output2, current_chunk_size);
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
            CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output2, current_chunk_size * sizeof(int64_t), cudaMemcpyDeviceToHost, stream2));
        }
    }

    // Synchronize streams to ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    double end = omp_get_wtime();

    // Free resources
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_vector1));
    CUDA_CHECK(cudaFree(d_vector2));
    CUDA_CHECK(cudaFree(d_output1));
    CUDA_CHECK(cudaFree(d_output2));
    free(h_vector);
    free(h_output);

    printf("gpu.c took %lf seconds\n", end-start);
    return EXIT_SUCCESS;
}
