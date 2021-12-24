#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

#define TPB_LIMIT 1024
#define MBPGD_LIMIT 65535
#define AUX_SIZE 1024000

/*THREAD PER BLOCK*/
static int global_tpb = 256; 
/*BLOCKS PER GRID*/
static int global_mbpg = 4096;


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

/*o que esta comentdo esta errado, nao esta preenchendo toda a matriz, farei de um modo mais simples*/

// __global__
// //void aux_func_matrix_mult(int n, float *a_d_rows, float *b_d_rows, float *c_d_rows, int width_a, int width_b, int width_c){
// void aux_func_matrix_mult(int n, int matrix_a_width,int matrix_a_height, int matrix_b_height, int matrix_b_width, float * matrix_a_rows, float * matrix_b_rows, float * matrix_c_rows){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 	for(int i = index; i < n; i+= stride){
// 		for(int k =0; k < matrix_a_width; k++){
// 			 matrix_c_rows[i] += matrix_a_rows[matrix_a_width*(i/matrix_a_height) + k] * matrix_b_rows[(i%matrix_a_height) + k*matrix_b_width];
//     }
//   }
//     /*for (aux_index; aux_index < n; aux_index += stride) {
// 		int line_for_c = aux_index/width_c;
//         int column_for_c = aux_index%width_c;
// 		int line_for_a = line_for_c*width_a;
// 		c_d_rows[aux_index] = 0.0;
//         int i=0;
//         int j=line_for_a;
// 		for(j; j < line_for_a + width_a; j++){
// 			c_d_rows[i] += a_d_rows[j] * b_d_rows[i*width_b + column_for_c];
// 			i++;
// 		}
//     }*/
// }

// int matrix_matrix_mult(Matrix *m_a, Matrix *m_b, Matrix *m_c){
// 	unsigned long int h= m_a->height;
//     unsigned long int width_a = m_b->width;
//     unsigned long int width_b = m_a->width;
// 	if((h%8!=0)||(width_a%8!=0)||(width_b%8!=0)){
//         printf("Erro: Tamanho das matrizes nao e divisivel por 8. \n");
// 		return 0;
// 	}
//     if(m_a == NULL || m_b == NULL || m_c ==NULL){
// 		printf("Erro: Uma ou mais matrizes não declaradas.\n");
// 		return 0;
// 	}
//     if(m_a->width != m_b->height){
// 		printf("Erro: A largura da matriz A precisa ser igual a altura da matriz B.\n");
// 		return 0;
// 	}
// 	int final_size=m_a->height*m_a->width;
// 	int loop_limit = (final_size + AUX_SIZE - 1) / AUX_SIZE;
// 	int chunk=AUX_SIZE;
// 	int i=0;
// 	for(i;i<chunk;i++){
// 		if(final_size%AUX_SIZE!=0 && i==loop_limit-1){
// 			chunk=final_size%AUX_SIZE;
// 		}
// 	}

// 	int block_size = global_tpb;
// 	int num_blocks = (chunk+ block_size - 1) / block_size;
// 	if (num_blocks > global_mbpg){
// 		num_blocks = global_mbpg;
// 		}

// 	//aux_func_matrix_mult<<<num_blocks, block_size>>>(chunk, m_a->d_rows, m_b->d_rows, m_c->d_rows, m_a->width, m_b->width, m_c->width);
	
// 	aux_func_matrix_mult<<<num_blocks, block_size>>>(chunk, m_a->width, m_a->height, m_b->height, m_b->width, m_a->d_rows, m_b->d_rows, m_c->d_rows);

// 	cudaDeviceSynchronize();
// 	return 1;
// }


__global__
void aux_func_matrix_mult(int h_a, int w_a, int h_b, int w_b, float * m_a_d, float * m_b_d, float * m_c_d){
    int i, j, index, stride;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    stride = gridDim.x * blockDim.x;

    for(i = index; i< h_a*w_b; i+= stride){
    	m_c_d[i] = 0;
	for(j = 0; j< w_a; j++){
		m_c_d[i] += m_a_d[w_a*(i/h_a) + j] * m_b_d[(i%h_a) + j*w_b];
	}
    }
}

int matrix_matrix_mult(struct matrix *m_a, struct matrix *m_b, struct matrix *m_c){
    int h_a = m_a->height;
    int w_a = m_a->width;
    int h_b = m_b->height;
    int w_b = m_b->width;
	if((h_a%8!=0)||(w_a%8!=0)||(w_b%8!=0)){
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
    int num_blocks = (m_c->height * m_c->width + global_tpb - 1) / global_tpb;
    aux_func_matrix_mult<<<num_blocks, global_tpb>>>(h_a, w_a, h_b, w_b, m_a->d_rows, m_b->d_rows, m_c->d_rows);
    return 1;
}
/*Essa função recebe o  número de threads  por bloco e o número máximo de blocos por grid 
que  devem  ser  usados  como  parâmetros  para  disparar  os  threads  (funções  kernel)  em 
paralelo  durante  o  processamento  das  operações  aritméticas  com  as  matrizes  e  deve  ser 
chamada pelo programa principal antes das outras funções. Caso não seja chamada, o valor 
default do número de threads por bloco do módulo é 256 e do número de blocos por grid é 
4096.*/
int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
	if(threads_per_block > TPB_LIMIT || max_blocks_per_grid > MBPGD_LIMIT){
		return 0;
	}
	else{
		global_tpb = threads_per_block;
		global_mbpg = max_blocks_per_grid;
		return 1;
	}
}