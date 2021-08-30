
#include "matrix_lib.h"


/*PARTE 1*/
/*Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o
produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
função deve retornar 0.*/
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}
	for ( i=0;i<matrix->height;i++){
		for( j=0;j<matrix->height;j++){
			matrix->row[i*matrix->width+j]*=scalar_value;
		}
	}
	return 1;
}


/*Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da
matriz A pela matriz B. O resultado da operação deve ser retornado na matriz C. Em caso
de sucesso, a função deve retornar o valor 1. Em caso de erro, a função deve retornar 0.*/
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC){
	if(matrixA == NULL || matrixB == NULL || matrixC ==NULL){
		printf("Erro: Uma ou mais matrizes não declaradas.");
	}
	return 0;
	//check size
	if(matrixA->width != matrixB->height){
		printf("Erro: Matriz A precisa ter a mesma largura da altura da matriz B.");
		return 0;
	}
//tamanho do somatorio de mult de elementos de A com B = matrixA width

	for(int i=0;i<matrixC->height;i++){
		for(int j=0;j<matrixC->width;j++){
			float aux=0;
			//c=a*b
			for(int iteracoes=0 ; iteracoes<matrixA->width;iteracoes++){
				float aux_AtoC = matrixA->rows[i*(matrixA->width)+iteracoes];

				float aux_BtoC = matrixB->rows[iteracoes*(matrixB->width)+j];
			
				aux+=aux_AtoC*aux_BtoC;
			}
			matrixC->rows[i*(matrixC->width)+j]=aux;

		}
	}
	return 1;
}



