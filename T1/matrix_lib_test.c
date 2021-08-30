#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>
#include "timer.h" 


Matrix * fillEmptyMatrix(Matrix *matrix){
	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}
	unsigned long int mH = matrix->height;
	unsigned long int mW = matrix->width;
	for(int i=0;i<m; i++){
		for(int j=0; j<n; j++){
			matrix->rows[i*n + j] = 0.0;
		}
	}
	return matrix;
}


void fillMatrixWithFile(Matrix matrix, char* filename){
	FILE *file;
	file = fopen (filename, "rb"); 
    if (file == NULL) { 
        fprintf(stderr, " - Erro: problema ao abrir arquivo\n"); 
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
	fwrite (&matrix.height, sizeof(unsigned long int), 1, file);

	for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fwrite (&matrix.rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	
   	fclose(file);

}



int main(int argc, char *argv[]){
	Matrix mA, mB, mC;
	float scalar = atof(argv[1]);

	/*INITIALIZE A*/
	unsigned long int linesForA = atoi(argv[2]);
	unsigned long int columnsForA = atoi(argv[3]);
	char* matrixAFile = argv[6];
	mA->height=linesForA;
	mA->width=columnsForA;
	mA->rows=(float*)malloc(mA.height*mA.width*sizeof(float));
	fillMatrixWtihFile(mA,matrixAFile);

	/*INITIALIZE B*/
	unsigned long int linesForB = atoi(argv[4]);
	unsigned long int columnsForB = atoi(argv[5]);
	char* matrixBFile = argv[7];
	mB->height=linesForB;
	mB->width=columnsForB;
	mB->rows=(float*)malloc(mB.height*mB.width*sizeof(float));
	fillMatrixWithFile(mB,matrixBFile);

	/*INITIALZE C*/
	mC->height=linesForA;
	mC->width=columnsForB;
	mC->rows=(float*)malloc(mC.height*mC.width*sizeof(float));
	fillEmptyMatrix(&mC);

	/*RESULTS*/
	char* firstResult = argv[8];
	char* secondResult = argv[9];



	/*SCALAR OF A*/
	printf("\n Scalar multiplication of Matrix A");
	scalar_matrix_mult(scalar,&mA);
	writeMatrixResult(mA,firstResult);

	/*SCALAR OF B*/
	//printf("\n Scalar multiplication of Matrix B");
	//scalar_matrix_mult(scalar,&mB);

	/*Matrix Multiplication*/
	printf("\n Matrix multiplication of Matrix A and Matrix B");
	matrix_matrix_mult(&mA,&mB,&mC);
	writeMatrixResult(mC,secondResult)

	return 0;
}