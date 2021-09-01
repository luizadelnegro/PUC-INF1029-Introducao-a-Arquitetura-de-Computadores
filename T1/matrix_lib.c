#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"



/*PARTE 1*/
/*Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o
produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
função deve retornar 0.*/
int scalar_matrix_mult(float scalar_value, Matrix *matrix){
	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}
	int max=matrix->width*matrix->height;
	for(int i=0;i<max;i++){
		matrix->rows[i]=matrix->rows[i]*scalar_value;
	}
/*	for ( int i=0;i<matrix->height;i++){
		for( int j=0;j<matrix->height;j++){
			matrix->rows[i*(matrix->width)+j]*=scalar_value;

		}
	}*/
	return 1;
}


/*Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da
matriz A pela matriz B. O resultado da operação deve ser retornado na matriz C. Em caso
de sucesso, a função deve retornar o valor 1. Em caso de erro, a função deve retornar 0.*/

int matrix_matrix_mult(Matrix *matrixA, Matrix * matrixB, Matrix * matrixC){
	
	if(matrixA == NULL || matrixB == NULL || matrixC ==NULL){
		printf("Erro: Uma ou mais matrizes não declaradas.");
		return 0;
	}

	//check size
	if(matrixA->width != matrixB->height){
		printf("Erro: A largura da matriz A precisa ser igual a altura da matriz B.");
		return 0;
	}
	
	for(int i=0 ; i<matrixA->height ; i++){
		for(int j=0 ; j<matrixB->width ; j++){
			int aux=0;
			for(int k=0 ; k<matrixA->width ; k++){
				//c=a*b
				float aux_AtoC = matrixA->rows[i*(matrixA->width)+j];
				float aux_BtoC = matrixB->rows[j*(matrixB->width)+k];
		
				aux+=aux_AtoC*aux_BtoC;
			}
			matrixC->rows[i*(matrixC->width)+j]=aux;
		}
	}
	
	return 1;
}

/*NOVA MM*/
int nova_matrix_matrix_mult(Matrix *matrixA, Matrix * matrixB, Matrix * matrixC){
	if(matrixA == NULL || matrixB == NULL || matrixC ==NULL){
		printf("Erro: Uma ou mais matrizes não declaradas.");
		return 0;
	}

	//check size
	if(matrixA->width != matrixB->height){
		printf("Erro: A largura da matriz A precisa ser igual a altura da matriz B.");
		return 0;
	}

	return 1;
}