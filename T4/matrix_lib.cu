#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

/*THREAD PER BLOCK*/
int global_tpb = 256; 
/*BLOCKS PER GRID*/
int global_mbpg = 4096;

/*PARTE 1*/
__global__
void aux_func_scalar(int n, float *d_rows, float scalar_value){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int aux_index=index;
    int stride = blockDim.x * gridDim.x;
    for (aux_index; aux_index < n; aux_index += stride) {
        d_rows[aux_index] = d_rows[aux_index] * scalar_value;
    }
}


int scalar_matrix_mult(float scalar_value, Matrix* matrix){
	unsigned long int h = matrix->height;
    unsigned long int w = matrix->width;	
    if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}
	if((h%8!=0)||(w%8!=0)){
		printf("Erro: Matriz com tamanho não divisivel por 8.");
		return 0;
	}
	int blocks = (h*w + global_tpb - 1) / global_tpb;
    /*garantir tamanho correto*/
	if (blocks > global_mbpg) {
        blocks = global_mbpg;
    }
	aux_func_scalar<<<blocks, global_tpb>>>(h*w, matrix->d_rows, scalar_value);
	cudaDeviceSynchronize();
	return 1;
}


__global__
void aux_func_matrix_mult(int n, float *a_d_rows, float *b_d_rows, float *c_d_rows, int width_a, int width_b, int width_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int aux_index=index;
    int stride = blockDim.x * gridDim.x;
    for (aux_index; aux_index < n; aux_index += stride) {
		int line_for_c = aux_index/width_c;
        int column_for_c = aux_index%width_c;
		int line_for_a = line_for_c*width_a;
		c_d_rows[aux_index] = 0.0;
        int i=0;
        int j=0;
		for(j = line_for_a; j < line_for_a + width_a; j++){
			c_d_rows[i] += a_d_rows[j] * b_d_rows[i*width_b + column_or_c];
			i++;
		}
    }
}

int matrix_matrix_mult(Matrix *m_a, Matrix *m_b, Matrix *m_c){
	unsigned long int h= m_a->height;
    unsigned long int width_a = m_b->width;
    unsigned long int width_b = m_a->width;
	if((h%8!=0)||(width_a%8!=0)||(width_b%8!=0)){
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
	int blocks = (matrixC->width*matrixC->height + global_tpb - 1) / global_tpb;
	if (blocks > global_mbpg) {
        blocks = global_mbpg;
    }
	aux_func_matrix_mult<<<blocks, global_tpb>>>(m_c->width*m_c->height, m_a->d_rows, m_b->d_rows, m_c->d_rows, m_a->width, m_b->width, m_c->width);
	cudaDeviceSynchronize();
	return 1;
}

/*Essa função recebe o  número de threads  por bloco e o número máximo de blocos por grid 
que  devem  ser  usados  como  parâmetros  para  disparar  os  threads  (funções  kernel)  em 
paralelo  durante  o  processamento  das  operações  aritméticas  com  as  matrizes  e  deve  ser 
chamada pelo programa principal antes das outras funções. Caso não seja chamada, o valor 
default do número de threads por bloco do módulo é 256 e do número de blocos por grid é 
4096.*/
int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
	if(threads_per_block > 1024 || mbpg > 65535){
		return 0;
	}
	else{
		global_tpb = threads_per_block;
		global_mbpg = max_blocks_per_grid;
		return 1;
	}
}