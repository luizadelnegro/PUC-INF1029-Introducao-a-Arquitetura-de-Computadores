typedef struct matrix {
unsigned long int height; //número de linhas da matriz (múltiplo de 8)
unsigned long int width; //número de colunas da matriz (múltiplo de 8)
float *rows; //sequência de linhas da matriz (height*width elementos)
} Matrix;

int scalar_matrix_mult(float scalar_value, Matrix *matrix);
int matrix_matrix_mult(Matrix *matrixA, Matrix * matrixB, Matrix * matrixC);