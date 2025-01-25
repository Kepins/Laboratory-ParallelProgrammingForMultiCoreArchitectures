#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 




int** generateRandomMatrix(int rows, int columns) {
    // Allocating memory for the matrix.
    int** matrix = (int**) malloc(rows * sizeof(int*));
    if (matrix == NULL) {  // Checking for unsuccessful memory allocation.
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }
    // Generating random numbers for each element in the matrix.
    srand(time(NULL));  // Seeding the random number generator with the current time.
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*) malloc(columns * sizeof(int));
        if (matrix[i] == NULL) {  // Checking for unsuccessful memory allocation.
            printf("Memory allocation failed.\n");
            exit(EXIT_FAILURE);  // Exiting the program with a failure status.
        }
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = rand() % 100;  // Generating a random number between 0 and 99.
        }
    }
    return matrix;  // Returning the generated matrix.
}

int** sumMatrices(int** matrix1, int** matrix2, int rows, int columns) {
    // Allocating memory for the result matrix.
    int** result = (int**) malloc(rows * sizeof(int*));
    if (result == NULL) {  // Checking for unsuccessful memory allocation.
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = (int*) malloc(columns * sizeof(int));
        if (result[i] == NULL) {  // Checking for unsuccessful memory allocation.
            printf("Memory allocation failed.\n");
            exit(EXIT_FAILURE);  // Exiting the program with a failure status.
        }
        for (int j = 0; j < columns; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return result;  // Returning the sum matrix.
}


int main(){
    int rows = 10000;
    int columns = 100000;

    // Generate two random matrices.
    int** matrix1 = generateRandomMatrix(rows, columns);
    int** matrix2 = generateRandomMatrix(rows, columns);
    
    double start = omp_get_wtime();

    // Sum the matrices.
    int** sum = sumMatrices(matrix1, matrix2, rows, columns);

    double end = omp_get_wtime();

    printf("Matrix addition took %lf seconds\n", (double)(end - start));

    // Free allocated memory to avoid memory leaks.
    for (int i = 0; i < rows; i++) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(sum[i]);
    }
    free(matrix1);
    free(matrix2);
    free(sum);


    return 0;
}