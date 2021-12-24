//processamento paralelo
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"
/*GRUPO 16*/
//ultima VE id 3(4a ve)

/*As alocações de memória no host e no device devem ser realizadas no programa principal, antes 
das chamadas das funções scalar_matrix_mult e matrix_matrix_mult*/
//programacontrolador e o q roda na ve
//implementacao das funcoes efetivamente

/*Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o 
produto do valor escalar pela matriz utilizando processamento vetorial e paralelo da AVEO.
Cada thread executada com processamento vetorial em uma Vector Engine (VE) deve 
calcular o resultado do produto entre o valor escalar e parte dos elementos da matriz em 
função da quantidade de threads paralelas. O resultado da operação deve ser retornado na 
matriz de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, 
a função deve retornar 0*/
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){

}

/*Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da 
matriz A pela matriz B utilizando processamento vetorial e paralelo da AVEO. Cada função 
thread executada com processamento vetorial em uma Vector Engine (VE) deve calcular o 
resultado de N linhas da matriz C, onde N é o número de linhas da matriz C dividido pelo 
número de threads disparadas em paralelo. O resultado da operação deve ser retornado na 
matriz C. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a função 
deve retornar 0.
*/
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC){

}

/*
Essa função recebe o número de identificação da Vector Engine (VE) na qual devem ser 
disparadas as threads em paralelo durante o processamento das operações aritméticas com 
as matrizes e deve ser chamada pelo programa principal antes das outras funções. O 
número máximo de Vector Engine (VE) disponíveis é 4 e os valores de num_node variam de 
0 a 3. Se a função for chamada com um valor de num_node inválido, o valor configurado 
deverá ser 0 (zero). Caso não seja chamada, o valor default 0 (zero) deve ser configurado.*/
void set_ve_execution_node(int num_node){

}

/*
Essa função recebe o número de threads que devem ser disparadas em paralelo durante o 
processamento das operações aritméticas com as matrizes e deve ser chamada pelo 
programa principal antes das outras funções. O número máximo de threads da Vector 
Engine (VE) da NEC é 8. Se a função for chamada com um valor superior a 8, o valor 
máximo do número de threads deve ser usado pelo módulo. Caso não seja chamada, o 
valor default do número de threads configurado no módulo deve ser 1*/
void set_number_threads(int num_threads){}


/*Essa função cria um processo na VE identificada por ve_execution_node e carrega a 
biblioteca dinâmica matrix_lib_ve.so no processo criado. Retorna 1 em caso de sucesso e 0 
(zero) em caso de erro na criação do processo ou erro no carregamento da biblioteca 
dinâmica.*/
int init_proc_ve_node(void){}
/*Essa função descarrega a biblioteca dinâmica matrix_lib_ve.so do processo criado na VE de 
execução e destrói o processo criado na VE de execução. Retorna 1 em caso de sucesso e 
0 (zero) em caso de erro no descarregamento da biblioteca dinâmica ou erro na destruição
do processo.*/
int close_proc_ve_node(void){}

/*Essa função carrega a matriz fornecida no processo criado na VE de execução. Para isso, a 
função aloca a memória necessária na VE e armazena o ponteiro no campo ve_rows da 
struct matrix. Depois, sincroniza os elementos de vh_rows com ve_rows. Retorna 1 em caso 
de sucesso e 0 (zero) em caso de erro alocação da memória na VE ou erro na sincronização
(cópia) dos elementos de vh_rows com ve_rows.*/
int load_ve_matrix(struct matrix *matrix){}

/*Essa função descarrega a matriz fornecida do processo criado na VE de execução. Para 
isso, a função sincroniza (copia) os elementos de ve_rows com vh_rows e, depois, libera a 
memória alocada na VE e armazena o valor NULL no campo ve_rows da struct matrix. 
Retorna 1 em caso de sucesso e 0 (zero) em caso de erro na sincronização (cópia) dos 
elementos de ve_rows com vh_rows ou erro na liberação da memória alocada na VE.*/
int unload_ve_matrix(struct matrix *matrix){}

/*Essa função sincroniza a memória do VH com a memória da VE fazendo a cópia dos 
elementos de vh_rows para ve_rows, previamente alocados com sucesso. Retorna 1 em 
caso de sucesso e 0 (zero) se vh_rows ou ve_rows forem NULL ou se ocorrer erro na cópia 
dos elementos de vh_rows para ve_rows.*/
int sync_vh_ve_matrix(struct matrix *matrix)
{}

/*Essa função sincroniza a memória da VE com a memória do VH fazendo a cópia dos 
elementos de ve_rows para vh_rows, previamente alocados com sucesso. Retorna 1 em 
caso de sucesso e 0 (zero) se vh_rows ou ve_rows forem NULL ou se ocorrer erro na cópia 
dos elementos de ve_rows para vh_rows.*/
int sync_ve_vh_matrix(struct matrix *matrix)
{}
