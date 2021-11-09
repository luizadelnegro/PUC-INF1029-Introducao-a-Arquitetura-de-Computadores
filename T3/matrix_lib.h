int number_of_threads;

typedef struct matrix {
unsigned long int height; //número de linhas da matriz (múltiplo de 8)
unsigned long int width; //número de colunas da matriz (múltiplo de 8)
float *rows; //sequência de linhas da matriz (height*width elementos)
} Matrix;

struct scalar_thread {
   int start;
   int end;
   int offset;
   int size;
   float scalar_value;
   struct matrix * matrix;
};

struct multiplication_thread{
	int start;
	int end;
    int offset;
    int size;
	struct matrix *A;
	struct matrix *B;
	struct matrix *C;
};


int scalar_matrix_mult(float scalar_value, Matrix *matrix);
int matrix_matrix_mult(Matrix *matrixA, Matrix * matrixB, Matrix * matrixC);
void set_number_threads(int n_threads);