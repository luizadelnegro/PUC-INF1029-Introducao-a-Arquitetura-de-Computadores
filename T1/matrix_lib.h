#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct matrix {
unsigned long int height; //número de linhas da matriz (múltiplo de 8)
unsigned long int width; //número de colunas da matriz (múltiplo de 8)
float *rows; //sequência de linhas da matriz (height*width elementos)
} Matrix;


Matrix * create_matrix(int matrix_height, int matrix_width);
int fill_matrix(float value, Matrix * matrix);
int show_matrix(Matrix * matrix, char * name);
int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);