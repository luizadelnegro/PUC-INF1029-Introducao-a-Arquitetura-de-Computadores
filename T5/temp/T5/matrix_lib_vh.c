#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
#include "matrix_lib.h"
/*GLOBAL*/
static int ve_execution_node = 0;
static int number_of_threads = 1;
static struct veo_proc_handle * process = NULL;
static uint64_t library_handle = 0;

/*PARTE 1*/
/*Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o 
produto do valor escalar pela matriz utilizando processamento vetorial e paralelo da AVEO.
Cada thread executada com processamento vetorial em uma Vector Engine (VE) deve 
calcular o resultado do produto entre o valor escalar e parte dos elementos da matriz em 
função da quantidade de threads paralelas. O resultado da operação deve ser retornado na 
matriz de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, 
a função deve retornar 0.*/
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {    
	int ret;
    struct veo_args *argp = veo_args_alloc();
	struct veo_thr_ctxt *ctx = veo_context_open(process);
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
    if (ctx == NULL) {
        printf("veo_context_open (scalar mult) failed\n");
        return 0;
    }
    if (argp == NULL) {
        printf("veo_args_alloc (scalar mult) failed\n");
        return 0;
    }
    veo_args_clear(argp);

    ret = veo_args_set_u64(argp, 0, matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_args_set_u64 failed for matrix_size: %d", ret);
        return 0;
    }
    ret = veo_args_set_hmem(argp, 1, matrix->ve_rows);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for ve_rows: %d", ret);
        return 0;
    }
    ret = veo_args_set_float(argp, 2, scalar_value);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for scalar: %d", ret);
        return 0;
    }
    ret = veo_args_set_i32(argp, 3, number_of_threads);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for num_threads: %d", ret);
        return 0;
    }
    uint64_t id = veo_call_async_by_name(ctx, library_handle, "scalar_mult", argp);
    if (id == VEO_REQUEST_ID_INVALID) {
        printf("veo_call_async_by_name (scalar mult) failed: %lu\n", id);
        return 0;
    }
    uint64_t retval;
    int wait_val;
    wait_val = veo_call_wait_result(ctx, id, &retval);
    if (wait_val != VEO_COMMAND_OK) {
        printf("veo_call_wait_result (scalar mult) failed: %d\n", wait_val);
        return 0;
    }
    veo_args_free(argp);
    ret = veo_context_close(ctx);
    if (ret != 0) {
        printf("veo_context_close (scalar mult) failed: %d\n", ret);
        return 0;
    }
    return 1;
}

/*Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da 
matriz A pela matriz B utilizando processamento vetorial e paralelo da AVEO. Cada função 
thread executada com processamento vetorial em uma Vector Engine (VE) deve calcular o 
resultado de N linhas da matriz C, onde N é o número de linhas da matriz C dividido pelo 
número de threads disparadas em paralelo. O resultado da operação deve ser retornado na 
matriz C. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a função 
deve retornar 0.*/
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
int matrix_matrix_mult(struct matrix *m_a, struct matrix *m_b, struct matrix *m_c){
    int h_a = matrixA->height;
    int w_a = matrixA->width;
    int h_b = matrixB->height;
    int w_b = matrixB->width;
	int ret;
    struct veo_args *argp = veo_args_alloc();
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
    struct veo_thr_ctxt *ctx = veo_context_open(process);
    if (ctx == NULL) {
        printf("veo_context_open (matrix mult) failed\n");
        return 0;
    }
    if (argp == NULL) {
        printf("veo_args_alloc (matrix mult) failed\n");
        return 0;
    }
    veo_args_clear(argp);
    ret = veo_args_set_u64(argp, 0, matrixC->height * matrixC->width);
    if (ret != 0) {
        printf("veo_args_set_u64 failed for height: %d", ret);
        return 0;
    }
    ret = veo_args_set_hmem(argp, 1, matrixC->ve_rows);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for ve_rows: %d", ret);
        return 0;
    }
    ret = veo_args_set_float(argp, 2, 1.0);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for scalar: %d", ret);
        return 0;
    }
    ret = veo_args_set_i32(argp, 3, number_of_threads);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for num_threads: %d", ret);
        return 0;
    }
    uint64_t id = veo_call_async_by_name(ctx, library_handle, "matrix_mult", argp);
    if (id == VEO_REQUEST_ID_INVALID) {
        printf("veo_call_async_by_name (matrix mult) failed: %lu\n", id);
        return 0;
    }
    uint64_t retval;
    int wait_val;
    wait_val = veo_call_wait_result(ctx, id, &retval);
    if (wait_val != VEO_COMMAND_OK) {
        printf("veo_call_wait_result (matrix mult) failed: %d\n", wait_val);
        return 0;
    }
    veo_args_free(argp);
    ret = veo_context_close(ctx);
    if (ret != 0) {
        printf("veo_context_close (matrix mult) failed: %d\n", ret);
        return 0;
    }
    return 1;
}


/*Essa função recebe o número de identificação da Vector Engine (VE) na qual devem ser 
disparadas as threads em paralelo durante o processamento das operações aritméticas com 
as matrizes e deve ser chamada pelo programa principal antes das outras funções. O 
número máximo de Vector Engine (VE) disponíveis é 4 e os valores de num_node variam de 
0 a 3. Se a função for chamada com um valor de num_node inválido, o valor configurado 
deverá ser 0 (zero). Caso não seja chamada, o valor default 0 (zero) deve ser configurado.*/
void set_ve_execution_node(int num_node) {
    if (num_node < 0 || num_node > 3) {
        ve_execution_node = 0;
        return;
    }
    ve_execution_node = num_node;
	return;
}
/*Essa função recebe o número de threads que devem ser disparadas em paralelo durante o 
processamento das operações aritméticas com as matrizes e deve ser chamada pelo 
programa principal antes das outras funções. O número máximo de threads da Vector 
Engine (VE) da NEC é 8. Se a função for chamada com um valor superior a 8, o valor 
máximo do número de threads deve ser usado pelo módulo. Caso não seja chamada, o 
valor default do número de threads configurado no módulo deve ser 1.*/
void set_number_threads(int num_threads) {
    if (num_threads < 1) {
        number_of_threads = 1;
       // printf("Warning: threads invalidas.\n", num_threads);
        return;
    }
    number_of_threads = num_threads <= 8 ? num_threads : 8;
	return;
}
/*Essa função cria um processo na VE identificada por ve_execution_node e carrega a 
biblioteca dinâmica matrix_lib_ve.so no processo criado. Retorna 1 em caso de sucesso e 0 
(zero) em caso de erro na criação do processo ou erro no carregamento da biblioteca 
dinâmica.
*/
int init_proc_ve_node() {
    process = veo_proc_create(ve_execution_node);
    if (process == NULL) {
        printf("Erro: veo_proc_create em init proc ve node\n");
        return 0;
    }

    library_handle = veo_load_library(process, "./matrix_lib_ve.so");

    if (library_handle == 0) {
        printf("Erro: veo_load_library em init_proc_ve_node\n");
        veo_proc_destroy(process);
        process = NULL;
        return 0;
    }
    return 1;
}
/*Essa função descarrega a biblioteca dinâmica matrix_lib_ve.so do processo criado na VE de 
execução e destrói o processo criado na VE de execução. Retorna 1 em caso de sucesso e 
0 (zero) em caso de erro no descarregamento da biblioteca dinâmica ou erro na destruição
do processo*/
int close_proc_ve_node() {
    int rc = veo_unload_library(process, library_handle);
    if (rc != 0) {
        printf("veo_unload_library failed: %d\n", rc);
        return 0;
    }
    library_handle = 0;
    rc = veo_proc_destroy(process);
    if (rc != 0) {
        printf("veo_proc_destroy failed: %d\n", rc);
        return 0;
    }
    process = NULL;
    return 1;
}

/*Essa função carrega a matriz fornecida no processo criado na VE de execução. Para isso, a 
função aloca a memória necessária na VE e armazena o ponteiro no campo ve_rows da 
struct matrix. Depois, sincroniza os elementos de vh_rows com ve_rows. Retorna 1 em caso 
de sucesso e 0 (zero) em caso de erro alocação da memória na VE ou erro na sincronização
(cópia) dos elementos de vh_rows com ve_rows*/
int load_ve_matrix(struct matrix *matrix) {
    int ret = veo_alloc_hmem(process, &(matrix->ve_rows), sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_alloc_hmem failed for ve_rows: %d\n", ret);
        return 0;
    }

    return sync_vh_ve_matrix(matrix);
}

/*Essa função descarrega a matriz fornecida do processo criado na VE de execução. Para 
isso, a função sincroniza (copia) os elementos de ve_rows com vh_rows e, depois, libera a 
memória alocada na VE e armazena o valor NULL no campo ve_rows da struct matrix. 
Retorna 1 em caso de sucesso e 0 (zero) em caso de erro na sincronização (cópia) dos 
elementos de ve_rows com vh_rows ou erro na liberação da memória alocada na VE.*/
int unload_ve_matrix(struct matrix *matrix) {
    if (sync_ve_vh_matrix(matrix) == 0) {
        printf("unload_ve_matrix failed when copying back\n");
        return 0;
    }

    int ret = veo_free_hmem(matrix->ve_rows);
    if (ret != 0) {
        printf("veo_free_hmem failed: %d\n", ret);
        return 0;
    }
    matrix->ve_rows = NULL;

    return 1;
}

static int null_element_check(Matrix * matrix) {
    if (matrix->vh_rows == NULL) {
        printf("vh_rows NULL!!!!\n");
        return 0;
    }
    if (matrix->ve_rows == NULL) {
        printf("ve_rows NULL!!!!\n");
        return 0;
    }
    return 1;
}
/*Essa função sincroniza a memória do VH com a memória da VE fazendo a cópia dos 
elementos de vh_rows para ve_rows, previamente alocados com sucesso. Retorna 1 em 
caso de sucesso e 0 (zero) se vh_rows ou ve_rows forem NULL ou se ocorrer erro na cópia 
dos elementos de vh_rows para ve_rows.
*/
int sync_vh_ve_matrix(struct matrix *matrix) {
    if (null_element_check(matrix) == 0) {
        printf("Error: found NULL when trying to copy (vh_rows -> ve_rows)\n");
        return 0;
    }
    int ret = veo_hmemcpy(matrix->ve_rows, matrix->vh_rows, sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_hmemcpy (vh_rows -> ve_rows) failed: %d\n", ret);
        return 0;
    }
    return 1;
}
/*Essa função sincroniza a memória da VE com a memória do VH fazendo a cópia dos 
elementos de ve_rows para vh_rows, previamente alocados com sucesso. Retorna 1 em 
caso de sucesso e 0 (zero) se vh_rows ou ve_rows forem NULL ou se ocorrer erro na cópia 
dos elementos de ve_rows para vh_rows.
*/
int sync_ve_vh_matrix(struct matrix *matrix) {
    if (verify_rows_for_null(matrix) == 0) {
        printf("Error: found NULL when trying to copy (ve_rows -> vh_rows)\n");
        return 0;
    }

    int ret = veo_hmemcpy(matrix->vh_rows, matrix->ve_rows, sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_hmemcpy (ve_rows -> vh_rows) failed: %d\n", ret);
        return 0;
    }
    return 1;
}
