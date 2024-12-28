#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ 
void calculate(unsigned long long input, int* dresults) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    int my_divisor = my_index + 2;

    if (input % my_divisor == 0 && my_divisor < input){
      dresults[my_index]=1;
    }
    else {
      dresults[my_index]=0;
    }
}


int main(int argc,char **argv) {

    int result;
    int threadsinblock=1024;
    int blocksingrid=10000;	

    int size = threadsinblock*blocksingrid;

    int *results;

    unsigned long long number;

    scanf("%Ld", &number);

    //unified memory allocation - available for host and device
    if (cudaSuccess!=cudaMallocManaged(&results,size*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(number, results);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //device synchronization to ensure that data in memory is ready
    cudaDeviceSynchronize();

    //calculate if any divides
    if (number == 0 || number == 1) {
      result = 0;
    }
    else {
      result=0;
      for(int i=0;i<size;i++) {
        result = result | results[i];
      }
      result = result == 1 ? 0 : 1;
    }
    printf("\nIs prime %d\n", result);

    //free memory
    if (cudaSuccess!=cudaFree(results))
      errorexit("Error when deallocating space on the GPU");

}
