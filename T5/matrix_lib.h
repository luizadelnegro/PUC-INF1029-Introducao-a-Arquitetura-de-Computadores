struct matrix {
unsigned long int height;//height = número de linhas da matriz (múltiplo de 8)
unsigned long int width;//width = número de colunas da matriz (múltiplo de 8)
float *vh_rows;//vh_rows = sequência de linhas da matriz (height*width elementos alocados no vector host)
void *ve_rows;//ve_rows = sequência de linhas da matriz (height*width elementos alocados na vector engine)

} ;



int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);
void set_ve_execution_node(int num_node);
void set_number_threads(int num_threads);
int init_proc_ve_node(void);
int close_proc_ve_node(void);
int load_ve_matrix(struct matrix *matrix);
int unload_ve_matrix(struct matrix *matrix);
int sync_vh_ve_matrix(struct matrix *matrix);
int sync_ve_vh_matrix(struct matrix *matrix);