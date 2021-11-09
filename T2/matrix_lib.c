#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"
#include <immintrin.h>

/*PARTE 1*/
int scalar_matrix_mult(float scalar_value, Matrix *matrix){
	unsigned long int h=matrix->height;
	unsigned long int w=matrix->width;
	__m256 scalar =_mm256_set1_ps(scalar_value);
	__m256 aux;
	__m256 scalar_result;


	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}

	if((h%8!=0)||(w%8!=0)){
		printf("Erro: Matriz com tamanho não divisivel por 8.");
		return 0;
	}
	/*float *rows_aux = matrix->rows;
	for(int i=0;i<w*h;i+=8,rows_aux+=8){
		aux = _mm256_load_ps(rows_aux);//carrega o item
		scalar_result=_mm256_mul_ps(aux, scalar);
		_mm256_store_ps(rows_aux, scalar_result);	
	}*/
	

	for(int i=0;i<h*w;i+=8){
		aux = _mm256_load_ps(&matrix->rows[i]);//carrega o item
		scalar_result=_mm256_mul_ps(aux, scalar);

		_mm256_store_ps(&matrix->rows[i], scalar_result);	

	}

	return 1;
}


int matrix_matrix_mult(Matrix *m_a, Matrix * m_b, Matrix * m_c){
	unsigned long int a_height = m_a->height;
	unsigned long int a_width = m_a->width;
	unsigned long int b_width = m_b->width;
	__m256 a_vector;
	__m256 b_vector;
	__m256 c_vector;
	__m256 escalar_a_b;
	float * a_next = m_a->rows;
	float * b_next = m_b->rows;
	float * c_next = m_c->rows;

	if((a_height%8!=0)||(a_width%8!=0)||(b_width%8!=0)){
		printf("Erro: Tamanho das matrizes nao e divisivel por 8. \n");
		return 0;
	}

	if(m_a == NULL || m_b == NULL || m_c ==NULL){
		printf("Erro: Uma ou mais matrizes não declaradas.\n");
		return 0;
	}

	if(m_a->width != m_b->height){
		printf("Erro: A largura da matriz A precisa ser igual a altura da matriz B.\n");
		return 0;
	}
//para todas as linhas de A
	for(int i = 0 ; i < a_height; i++, a_next+= 8){
		//a linha de c tem que ser a mesma de A
		c_next=m_c->rows+(i*b_width);

		for(int j = 0;j<a_width;j++){
			a_vector=_mm256_set1_ps(a_next[j]);

			for(int k = 0; k< b_width;k+=8,c_next+=8){
				if(j==0){
					c_vector=_mm256_set1_ps(0);//acho q n precisa dessa linha

				}
				else{
					c_vector=_mm256_load_ps(c_next);
				}
				b_vector=_mm256_load_ps(b_next);

				escalar_a_b=_mm256_fmadd_ps(a_vector, b_vector, c_vector);
				_mm256_store_ps(c_next, escalar_a_b);				
			}
			c_next= m_c->rows+(i*b_width);
		}
		b_next = m_b->rows;
	}

	return 1;
}

