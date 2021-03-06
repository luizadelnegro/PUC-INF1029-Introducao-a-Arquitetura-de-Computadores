#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>
#include "timer.h"

/*PARTE 2*/
/*
Crie um programa em linguagem C, chamado matrix_lib_test.c, que implemente um código para
testar a biblioteca matrix_lib.c.
[...]
O programa base principal deve cronometrar o tempo de execução geral do programa (overall time)
e o tempo de execução das funções scalar_matrix_mult e matrix_matrix_mult. Para marcar o início
e o final do tempo em cada uma das situações, deve-se usar a função padrão gettimeofday
disponível em <sys/time.h>. Essa função trabalha com a estrutura de dados struct timeval definida
em <sys/time.h>. Para calcular a diferença de tempo (delta) entre duas marcas de tempo t0 e t1,
deve-se usar a função timedifference_msec, implementada no módulo timer.c, fornecido abaixo:*/

void fillEmptyMatrix(Matrix *matrix){
	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return ;
	}
	unsigned long int mH = matrix->height;
	unsigned long int mW = matrix->width;
	for(int i=0;i<mH; i++){
		for(int j=0; j<mW; j++){
			matrix->rows[i*mW + j] = 0.0;
		}
	}
	return ;
}


void fillMatrixWithFile(Matrix matrix, char* filename){
	FILE *file;
	file = fopen (filename, "rb"); 
    if (file == NULL) { 
        fprintf(stderr, " - Erro: problema ao abrir arquivo \n"); 

        exit (1); 
    }

    fread(&matrix.width, sizeof(unsigned long int), 1, file);
    fread(&matrix.height, sizeof(unsigned long int), 1, file);
    for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fread (&matrix.rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	fclose(file);
}


void writeMatrixResult(Matrix matrix, char*filename){
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

void showMatrix(Matrix matrix){

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
	gcc -Wall -o test matrix_lib.c matrix_lib.h matrix_lib_test.c
	./test 5.0 8 16 16 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat


	gcc -Wall -o test matrix_lib_test.c matrix_lib.c matrix_lib.h timer.c timer.h
	./test 5.0 1024 1024 1024 1024 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat

	*/

	/*TIME*/
	struct timeval start, stop, startOverall, stopOverall;
	gettimeofday(&startOverall, NULL);


	float scalar = atof(argv[1]);
	/*MATRIX*/
	unsigned long int linesForA = atoi(argv[2]);
	unsigned long int columnsForA = atoi(argv[3]);
	unsigned long int linesForB = atoi(argv[4]);
	unsigned long int columnsForB = atoi(argv[5]);
	char* matrixAFile = argv[6];
	char* matrixBFile = argv[7];

	/*RESULTS*/
	char* firstResult = argv[8];
	char* secondResult = argv[9];
	
	Matrix mA;
	Matrix mB;
	Matrix mC;

	/*INITIALIZE A*/
	mA.height=linesForA;
	mA.width=columnsForA;
	mA.rows=(float*)malloc(mA.height*mA.width*sizeof(float));
	gettimeofday(&start, NULL);
	fillMatrixWithFile(mA,matrixAFile);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of filling Matrix A with file: %f ms\n",timedifference_msec(start, stop));

	/*INITIALIZE B*/
	mB.height=linesForB;
	mB.width=columnsForB;
	mB.rows=(float*)malloc(mB.height*mB.width*sizeof(float));
	fillMatrixWithFile(mB,matrixBFile);

	/*INITIALZE C*/
	mC.height=linesForA;
	mC.width=columnsForB;
	mC.rows=(float*)malloc(mC.height*mC.width*sizeof(float));
	gettimeofday(&start, NULL);
	fillEmptyMatrix(&mC);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of filling Matrix C with 0: %f ms\n",timedifference_msec(start, stop));

	/*SCALAR OF A*/
	gettimeofday(&start, NULL);
	scalar_matrix_mult(scalar,&mA);
	gettimeofday(&stop, NULL);
	writeMatrixResult(mA,firstResult);
	printf("\n Time difference of scalar multiplicaton of Matrix A: %f ms\n",timedifference_msec(start, stop));

	/*Matrix Multiplication*/
	gettimeofday(&start, NULL);
	matrix_matrix_mult(&mA,&mB,&mC);
	gettimeofday(&stop, NULL);
	printf("\n Time difference of multiplicaton of Matrix A and Matrix B: %f ms\n",timedifference_msec(start, stop));
	writeMatrixResult(mC,secondResult);

	/*Optimized Matrix Multiplication*/
	gettimeofday(&start,NULL);
	nova_matrix_matrix_mult(&mA,&mB,&mC);
	gettimeofday(&stop,NULL);
	printf("\n Time difference of the optimized multiplicaton of Matrix A and Matrix B: %f ms\n",timedifference_msec(start, stop));
	
	gettimeofday(&stopOverall, NULL);
	printf("Overall time: %f ms\n", timedifference_msec(startOverall, stopOverall));
	return 0;
}