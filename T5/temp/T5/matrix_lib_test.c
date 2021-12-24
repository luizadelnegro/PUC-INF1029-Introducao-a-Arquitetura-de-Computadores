#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "matrix_lib.h"
#include "timer.h"
 


static void fill_matrix_with_file(struct matrix *matrix, const char * filename) {
    FILE *file;
    file = fopen(filename, "rb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo.\n");
        system("pause");
        exit(1);
    }
    fread(matrix->vh_rows, matrix->height * matrix->width * sizeof(float), 1, file);
    fclose(file);

    if (load_ve_matrix(matrix) == 0) {
        printf("load_ve_matrix (matrix.vh_rows -> matrix.ve_rows) filling with file %s returned error\n", fileName);
        exit(1);
    }
}


void fill_matrix_with_value(struct matrix  * matrix, float value){
	unsigned long int h = matrix->height;
    unsigned long int w = matrix->width;
	for(int i=0;i<h*w; i++){
		matrix->h_rows[i] = value;
	}
	if (load_ve_matrix(matrix) == 0) {
        printf("Erro: Problema na alocação da matriz C\n");
        exit(1);
    }
}

void show_matrix(Matrix matrix){
	printf("[ ");
	for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			printf(" %.1f ",matrix.h_rows[i*matrix.width + j]);
		}
		printf("\n");
	}
	printf("]\n");
}


    

int main(int argc, char* argv[]) 
/*Como rodar 
   ./matrix_lib_test 5.0 8 16 16 8 256 4096 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat

*/
/*TIME*/
	struct timeval start, stop, startOverall, stopOverall;
	gettimeofday(&startOverall, NULL);

	float scalar = strtof(argv[1],NULL);
	/*MATRIX*/
	//TO DO: conferir se nao da pra usar  atoi(argv[2]);
	unsigned long int lines_for_a = strtoul(argv[2],NULL, 10);
	unsigned long int columns_for_a = strtoul(argv[3],NULL, 10);
	unsigned long int lines_for_b = strtoul(argv[4],NULL, 10);
	unsigned long int columns_for_b = strtoul(argv[5],NULL, 10);
	// ate aqui
	int ve_identifier =atoi(argv[6]);
	int number_of_threads = atoi(argv[7]);
	char* matrix_a_file = argv[8];
	char* matrix_b_file = argv[9];
	/*RESULTS*/
	char* first_result = argv[10];
	char* second_result = argv[11];
	
	Matrix matrix_a;
	Matrix matrix_b;
	Matrix matrix_c;

/*start ve exec first and setup number of threads*/
    set_ve_execution_node(ve_identifier);
    set_number_threads(number_of_threads);
    init_proc_ve_node();

	/*INITIALIZE A*/
	matrix_a.height=lines_for_a;
	matrix_a.width=columns_for_a;
	matrix_a.vh_rows = (float *) malloc(matrix_a.height * matrix_a.width * (sizeof(float)));
	fill_matrix_with_file(matrix_a,matrix_a_file);

	/*INITIALIZE B*/
	matrix_b.height = lines_for_b;
	matrix_b.width = columns_for_b;
	matrix_b.vh_rows = (float *) malloc(matrix_b.height * matrix_b.width * (sizeof(float)));
	fill_matrix_with_file(matrix_b,matrix_b_file);
	
	/*INITIALZE C*/
	matrix_c.height = lines_for_a;
	matrix_c.width = columns_for_b; 
	matrix_c.vh_rows = (float *) malloc(matrix_c.height * matrix_c.width * (sizeof(float)));
	//TO DO func abaixo
	fill_matrix_with_value(&matrix_c,0);
	
	/*SCALAR OF A*/
	gettimeofday(&start, NULL);
	int scalar_result=scalar_matrix_mult(scalar,matrix_a);
	gettimeofday(&stop, NULL);
	if (scalar_result==0){
        printf("Erro: Problema na multiplicação por escalar \n");
    } else {
		printf("\n Time difference of scalar multiplicaton of Matrix A: %f ms\n",timedifference_msec(start, stop));
        if (sync_ve_vh_matrix(matrix_a) == 0) {
            printf("Erro: Problema ao salvar matriz.\n");
        } else {
			fill_file_with_matrix(matrix_a,first_result);
        }
    }
	/*MATRIX MULT*/
    gettimeofday(&start, NULL);
    int matrix_mult_result = matrix_matrix_mult(matrix_a, matrix_b, matrix_c);
    gettimeofday(&stop, NULL);

    if (matrix_mult_result == 0) {
        printf("Erro: Problema na multiplicação entre matrizes\n");
    } else {
		printf("\n Time difference of multiplicaton of Matrix A and Matrix B: %f ms\n",timedifference_msec(start, stop));
        if (unload_ve_matrix(matrix_c) == 0) {
            printf("Erro ao copiar a matriz C antes de salvar.\n");
        } else {
            fill_file_with_matrix(matrix_c,second_result);
        }
    }

    /*FREE*/


	close_proc_ve_node();

	gettimeofday(&stopOverall, NULL);
	printf("Overall time: %f ms\n", timedifference_msec(startOverall, stopOverall));

    return 0;
}