#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>


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
	fillMatrixWithFile(mA,matrixAFile);

	/*INITIALIZE B*/
	mB.height=linesForB;
	mB.width=columnsForB;
	mB.rows=(float*)malloc(mB.height*mB.width*sizeof(float));
	fillMatrixWithFile(mB,matrixBFile);

	/*INITIALZE C*/
	mC.height=linesForA;
	mC.width=columnsForB;
	mC.rows=(float*)malloc(mC.height*mC.width*sizeof(float));
	fillEmptyMatrix(&mC);




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
	writeMatrixResult(mC,secondResult);

	return 0;
}