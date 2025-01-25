#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MAX_RECURSION_DEPTH_PARALLEL 10

void generateRandomNumbers(int *arr, int N) {
	srand(time(NULL));

  const int A = 0;
  const int B = 100000;

  for (int i = 0; i < N; i++) {
    arr[i] = A + rand() % (B - A + 1);
  }
}

void swap(int* a, int* b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

int partition(int arr[], int low, int high) {
  int p = arr[low];
  int i = low;
  int j = high;

  while (i < j) {
    while (arr[i] <= p && i <= high - 1) {
      i++;
    }
    while (arr[j] > p && j >= low + 1) {
      j--;
    }
    if (i < j) {
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[low], &arr[j]);
  return j;
}

void quickSort(int arr[], int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);

    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

void quickSortParralel(int arr[], int low, int high, int depth) {
  if (low < high) {
    int pi = partition(arr, low, high);

    if (depth < MAX_RECURSION_DEPTH_PARALLEL){
      #pragma omp parallel num_threads(2)
      {
        int my_id = omp_get_thread_num();
        if(my_id==0){
          quickSortParralel(arr, low, pi - 1, depth+1);
        }
        else {
          quickSortParralel(arr, pi + 1, high, depth+1);
        }
      }
    }
    else {
      quickSortParralel(arr, low, pi - 1, depth+1);
      quickSortParralel(arr, pi + 1, high, depth+1);
    }
  }
}


int main(int argc,char **argv) {
  int N;
  double sequential, parallel;

  omp_set_nested(1); //enables nested parallelism
  omp_set_dynamic(0); //disable dynamic setting for number of threads

  printf("Enter number of elements: \n");
  scanf("%d", &N);

	int *arr = (int *)malloc(N * sizeof(int));
  if (arr == NULL) {
    printf("Memory allocation failed.\n");
   return 1;
  }
  int *arr2 = (int *)malloc(N * sizeof(int));
  if (arr2 == NULL) {
    printf("Memory allocation failed.\n");
   return 1;
  }
	generateRandomNumbers(arr, N);
  for(int i=0;i<N;i++){
    arr2[i] = arr[i];
  }

  double start_s = omp_get_wtime();
  quickSort(arr, 0, N);
  double end_s = omp_get_wtime();
  sequential = end_s-start_s;

  double start_p = omp_get_wtime();
  quickSortParralel(arr2, 0, N, 0);
  double end_p = omp_get_wtime();
  parallel = end_p-start_p;

  printf("QuickSort sequential took %lf seconds\n", sequential);
  printf("QuickSort parallel took %lf seconds\n", parallel);
}
