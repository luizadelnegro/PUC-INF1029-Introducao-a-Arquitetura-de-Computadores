#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>
#include <immintrin.h>
#include "timer.h"


void fill_empty_matrix(Matrix matrix){
	__m256 value = _mm256_set1_ps(0);
	for(int i=0; i<matrix.height*matrix.width; i+=8){
		_mm256_store_ps(&matrix.rows[i], value);
	}
	return;
}

void fill_matrix_with_file(Matrix matrix, char* filename){
	FILE *file;
	file = fopen (filename, "rb"); 
	float aux_value = 0;
	__m256 value;
  	if (file == NULL) { 
        printf(" - Erro: problema ao abrir arquivo \n"); 

        exit (1); 
    }
    
    fread(&matrix.width, sizeof(unsigned long int), 1, file);// tem que ver se o arquivo vai ter os 2 primeiros numeros o tamanho da matriz
    fread(&matrix.height, sizeof(unsigned long int), 1, file);
    int w=matrix.width;
    int h=matrix.height;
    for(int i=0;i<h*w;i+=8){
		fread((void*)(&aux_value), sizeof(float), 1, file);
		value = _mm256_set1_ps(aux_value);
    	_mm256_store_ps(&matrix.rows[i], value);
    }

	fclose(file);
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

void fill_file_with_matrix(Matrix matrix, char*filename){
	FILE *file;
	file = fopen (filename, "wb");
	if (file == NULL) { 
        fprintf(stderr, "- Erro: problema ao abrir arquivo \n"); 
        exit (1); 
    }
	
    fwrite (&matrix.width, sizeof(unsigned long int), 1, file);

	for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fwrite (&matrix.rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	
   	fclose(file);

}






int main (int argc, char *argv[]){
/*
	Como rodar:
	gcc -Wall –std=c11 –mfma -o test matrix_lib.c matrix_lib.h matrix_lib_test.c timer.h timer.c
	*/

	/*TIME*/
	struct timeval start, stop, startOverall, stopOverall;
	gettimeofday(&startOverall, NULL);

	float scalar = atof(argv[1]);
	/*MATRIX*/
	unsigned long int lines_for_a = atoi(argv[2]);
	unsigned long int columns_for_a = atoi(argv[3]);
	unsigned long int lines_for_b = atoi(argv[4]);
	unsigned long int columns_for_b = atoi(argv[5]);
	char* matrix_a_file = argv[6];
	char* matrix_b_file = argv[7];

	/*RESULTS*/
	char* first_result = argv[8];
	char* second_result = argv[9];

	Matrix matrix_a;
	Matrix matrix_b;
	Matrix matrix_c;

	/*INITIALIZE A*/
	matrix_a.height=lines_for_a;
	matrix_a.width=columns_for_a;
	float* a_vector= (float*)aligned_alloc(32, (matrix_a.height*matrix_a.width) * sizeof(float));
	matrix_a.rows=a_vector;
	gettimeofday(&start, NULL);
	fill_matrix_with_file(matrix_a,matrix_a_file);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of filling Matrix A with file: %f ms\n",timedifference_msec(start, stop));


	/*INITIALIZE B*/
	matrix_b.height = lines_for_b;
	matrix_b.width = columns_for_b;
	float* b_vector = (float*)aligned_alloc(32, (matrix_b.height*matrix_b.width) * sizeof(float));
	matrix_b.rows = b_vector;
	fill_matrix_with_file(matrix_b,matrix_b_file);


	/*INITIALZE C*/
	matrix_c.height = lines_for_a;
	matrix_c.width = columns_for_b; 
	float* c_vector = (float*)aligned_alloc(32, (matrix_c.height*matrix_c.width) * sizeof(float));
	matrix_c.rows = c_vector;
	gettimeofday(&start, NULL);
	fill_empty_matrix(matrix_c);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of filling Matrix C with 0: %f ms\n",timedifference_msec(start, stop));


	/*SCALAR OF A*/
	gettimeofday(&start, NULL);
	scalar_matrix_mult(scalar,&matrix_a);
	gettimeofday(&stop, NULL);
	fill_file_with_matrix(matrix_a,first_result);
	printf("\n Time difference of scalar multiplicaton of Matrix A: %f ms\n",timedifference_msec(start, stop));

	/*MATRIX MULTIPLICATION */
	gettimeofday(&start, NULL);
	matrix_matrix_mult(&matrix_a, &matrix_b, &matrix_c);
	gettimeofday(&stop, NULL);
	fill_file_with_matrix(matrix_c,second_result);
	printf("\n Time difference of multiplicaton of Matrix A and Matrix B: %f ms\n",timedifference_msec(start, stop));
	
	free(a_vector);
	free(b_vector);
	free(c_vector);

	gettimeofday(&stopOverall, NULL);
	printf("Overall time: %f ms\n", timedifference_msec(startOverall, stopOverall));


	return 0;
}