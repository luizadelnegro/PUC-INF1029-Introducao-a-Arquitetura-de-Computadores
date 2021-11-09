#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <cuda_runtime.h>
extern "C" {
  #include "timer.h"
}
#include "matrix_lib.h"


/*PARTE 2*/
void fill_file_with_matrix(Matrix matrix, char *filename){
	FILE *file;
	file = fopen (filename, "wb");
	if (file == NULL) { 
        fprintf(stderr, " - Erro: problema ao abrir arquivo \n"); 
        exit (1); 
    }
	
    fwrite (&matrix.width, sizeof(unsigned long int), 1, file);
	fwrite (&matrix.height, sizeof(unsigned long int), 1, file);

	for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fwrite (&matrix.h_rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	
   	fclose(file);
}

void fill_matrix_with_file(Matrix matrix, char *filename){
	FILE *file;
	file = fopen (filename, "rb"); 
 	if (file == NULL) { 
        printf(" - Erro: problema ao abrir arquivo \n"); 

        exit (1); 
    }

    fread(&matrix.width, sizeof(unsigned long int), 1, file);
    fread(&matrix.height, sizeof(unsigned long int), 1, file);
    for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fread (&matrix.h_rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	fclose(file);

}

void fill_matrix_with_value(Matrix matrix, float value){
	unsigned long int h = matrix->height;
    unsigned long int w = matrix->width;
	for(int i=0;i<h*w; i++){
		matrix->h_rows[i] = value;
	}
}


int check_errors(Matrix *matrix, float scalar_value) {
    unsigned long int i;
    unsigned long int num_elements_matrix = matrix->height * matrix->width;
  /* Check the integrity of the matrix */
    if (num_elements_matrix == 0){
        print("Erro: matriz sem dimensoes \n")
        return 0;
    }
    if(matrix->h_rows == NULL) {
        print("Erro: matrix Null\n")
        return 0;
    }
    float max_error_value = 0.0f;
    float aux_error_value = 0.0f;
    for (i = 0; i < num_elements_matrix; i++){
        max_error_value = (max_error_value > (aux_error_value=fabs(matrix->h_rows[i]-scalar_value)))? max_error_value : aux_error_value;
    }
    printf("Quantidade de erros =   %f\n", max_error_value);
    return 1;
}

void show_matrix(Matrix matrix){

	unsigned long int mH = matrix.height;
	unsigned long int mW = matrix.width;

	printf("[ ");
	for(int i=0;i<mH; i++){
		for(int j=0; j<mW; j++){
			printf(" %.1f ",matrix.rows[i*mW + j]);
		}
		printf("\n");
	}
	printf("]\n");


}

int main(int argc, char *argv[]){
/*
	Como rodar:
	gcc -Wall –std=c11 –mfma -o test matrix_lib.c matrix_lib.h matrix_lib_test.c timer.h timer.c
	
	gcc -std=c11 -pthread -mfma -o test matrix_lib_test.c matrix_lib.c timer.c
	*/

	/*TIME*/
	struct timeval start, stop, startOverall, stopOverall;
	gettimeofday(&startOverall, NULL);

    /*CUDA*/
	cudaError_t cuda_error;

    float scalar = atof(argv[1]);
	/*MATRIX*/
	unsigned long int lines_for_a = atoi(argv[2]);
	unsigned long int columns_for_a = atoi(argv[3]);
	unsigned long int lines_for_b = atoi(argv[4]);
	unsigned long int columns_for_b = atoi(argv[5]);
	int n_threads = atoi(argv[6]);
	int max_blocks = atoi(argv[7]);
    char* matrix_a_file = argv[8];
	char* matrix_b_file = argv[9];
	/*RESULTS*/
	char* first_result = argv[10];
	char* second_result = argv[11];


    /*SET GRID SIZE FUNCTION*/
	set_grid_size(num_threads, max_blocks);
	

    /*DECLARE MATRIX*/
    Matrix matrix_a;
	Matrix matrix_b;
	Matrix matrix_c;


    /*OBSERVACAO 2 - As matrizes A, B e C devem ser alocadas simultaneamente e por completo na memória da GPGPU 
NVIDIA Tesla C2075 que tem 5GB de memória disponível. Se não for viável fazer a alocação, o 
programa principal deve emitir uma notificação de erro de alocação de memória para o usuário. */
	/*INITIALIZE A */ 
	matrix_a.height=lines_for_a;
	matrix_a.width=columns_for_a;
	matrix_a.h_rows = (float*)aligned_alloc(32,matrix_a.height*matrix_a.width*sizeof(float));

	cuda_error = cudaMalloc(&matrix_a.d_rows, matrix_a.height*matrix_a.width*sizeof(float));
	if (cuda_error != cudaSuccess) {
		printf("Erro %s: malloc matrix A - codigo %d\n", cudaGetErrorString(cuda_error), cuda_error);
			return 1;
	}

    fill_matrix_with_file(matrix_a,matrix_a_file)

    cuda_error = cudaMemcpy(matrix_a.d_rows, matrix_a.h_rows, matrix_a.height*matrix_a.width*sizeof(float), cudaMemcpyHostToDevice);

	if (cuda_error != cudaSuccess) {
		printf("Erro %s: cudaMemcpy matrix A  - codigo %d - linha %d \n", cudaGetErrorString(cuda_error), cudaError, __LINE__);
			return 1;
	}


    /*INITIALIZE B*/
	matrix_b.height = lines_for_b;
	matrix_b.width = columns_for_b;
	matrix_b.h_rows = (float*)aligned_alloc(32,matrix_b.height*matrix_b.width*sizeof(float));
	cuda_error = cudaMalloc(&B.d_rows, B.height*B.width*sizeof(float));
	if (cuda_error != cudaSuccess) {
		printf("Erro %s: malloc matrix B - codigo %d\n", cudaGetErrorString(cuda_error), cuda_error);
			return 1;
	}

    fill_matrix_with_file(matrix_b,matrix_b_file)

	cuda_error = cudaMemcpy(matrix_b.d_rows, matrix_b.h_rows, matrix_b.height*matrix_b.width*sizeof(float), cudaMemcpyHostToDevice);

	if (cuda_error != cudaSuccess) {
		printf("Erro %s: cudaMemcpy matrix B - codigo %d - linha %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}

    /*INITIALZE C */
	matrix_c.height = lines_for_a;
	matrix_c.width = columns_for_b; 
	matrix_c.h_rows = (float*)aligned_alloc(32,matrix_c.height*matrix_c.width*sizeof(float));

	cuda_error = cudaMalloc(&matrix_c.d_rows, matrix_c.height*matrix_c.width*sizeof(float));
	if (cuda_error != cudaSuccess) {
		printf("Erro %s: malloc matrix C - codigo: %d\n", cudaGetErrorString(cuda_error), cuda_error);
			return 1;
	}
	

    /*SCALAR MULT OF A*/
	gettimeofday(&start, NULL);
	scalar_matrix_mult(scalar, &matrix_a);
	gettimeofday(&stop, NULL);
	cuda_error = cudaMemcpy(matrix_a.h_rows, matrix_a.d_rows, matrix_a.height*matrix_a.width*sizeof(float), cudaMemcpyDeviceToHost);
	if (cuda_error != cudaSuccess) {
		printf("Erro %s: SCALAR FUNC - cudaMemcpy matrix A - codigo %d - linha %d\n", cudaGetErrorString(cuda_error), cuda_error, __LINE__);
			return 1;
	}
    printf("\n Time difference of scalar multiplicaton of Matrix A: %f ms\n",timedifference_msec(start, stop));
	printf(" Erros na scalar mult: ");
  	check_errors(&matrix_a, 20.0f);
    printf("\n")
    fill_file_with_matrix(matrix_a,first_result)

    show_matrix(&first_result)



    /*MATRIX MULT*/
	gettimeofday(&start, NULL);	
	matrix_matrix_mult(&matrix_a, &matrix_b, &matrix_c);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of multiplicaton of Matrix A and Matrix B: %f ms\n",timedifference_msec(start, stop));
	
	cuda_error = cudaMemcpy(matrix_c.h_rows, matrix_c.d_rows, matrix_c.height*matrix_c.width*sizeof(float), cudaMemcpyDeviceToHost);

	if (cuda_error != cudaSuccess) {
		printf("Erro %s: MATRIX MULT - cudaMemcpy Matrix C - codigo %d - linha %d \n", cudaGetErrorString(cuda_error), cuda_error, __LINE__);
			return 1;
	}

    printf(" Matrix C  - Erros na matrix mult ");
  	check_errors(&matrix_c, 640.0f);
    printf("\n")
    fill_file_with_matrix(matrix_c,second_result);





    free(matrix_a.h_rows);
	free(matrix_b.h_rows);
	free(matrix_c.h_rows);


	cudaFree(matrix_a.d_rows);
	cudaFree(matrix_b.d_rows);
	cudaFree(matrix_c.d_rows);

   	gettimeofday(&stopOverall, NULL);
	printf("Overall time: %f ms\n", timedifference_msec(startOverall, stopOverall));


    return 0; 
} 