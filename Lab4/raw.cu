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
    int blocksingrid=1000;	

    int size = threadsinblock*blocksingrid;


    unsigned long long number;

    scanf("%Ld", &number);

    //memory allocation on host
    int *hresults=(int*)malloc(size*sizeof(int));
    if (!hresults) errorexit("Error allocating memory on the host");	

    int *dresults=NULL;

    //devie memory allocation (GPU)
    if (cudaSuccess!=cudaMalloc((void **)&dresults,size*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(number, dresults);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //getting results from GPU to host memory
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,size*sizeof(int),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");

    //calculate if any divides
    if (number == 0 || number == 1) {
      result = 0;
    }
    else {
      result=0;
      for(int i=0;i<size;i++) {
        result = result | hresults[i];
      }
      result = result == 1 ? 0 : 1;
    }
    printf("\nIs prime %d\n", result);

    //free memory
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");
}
