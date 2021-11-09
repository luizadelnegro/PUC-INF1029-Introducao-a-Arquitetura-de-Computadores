#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

typedef struct matrix{
	unsigned long int height;
	unsigned long int width;
	float *h_rows;
	float *d_rows;
} Matrix;

int scalar_matrix_mult(float scalar_value, Matrix *matrix);

int matrix_matrix_mult(Matrix *ma, Matrix *mb, Matrix *mc);

int set_grid_size(int threads_per_block, int max_blocks_per_grid);