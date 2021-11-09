#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_lib.h"
#include <immintrin.h>
#include <pthread.h>

/*PARTE 1*/
void set_number_threads(int n_threads){
	if (n_threads > 0){
		number_of_threads = n_threads;
	}
	else{
		number_of_threads = 1;
	}
}

void * avx_scalar(void * arg){
	__m256 scalar;
	__m256 aux;
	__m256 scalar_result;
	struct scalar_thread * current;
	current=(struct scalar_thread*)arg;
	int end = (current->size / number_of_threads) + current->offset;
  	if(current->end){
    	end = current->size;
  	}
	float *next= &current->matrix->rows[current->offset];
	for(int i=current->offset;i<end;i+=8,next+=8){
		aux = _mm256_load_ps(next);//carrega o item
		scalar = _mm256_set1_ps(current->scalar_value);
		scalar_result=_mm256_mul_ps(aux, scalar);
		_mm256_store_ps(next, scalar);	
	}
  	pthread_exit(NULL);
}

int scalar_matrix_mult(float scalar_value, Matrix *matrix){
	int thread_created;
	void * status;
	pthread_attr_t attr;
	pthread_t threads[number_of_threads];
	struct scalar_thread thread_array[number_of_threads];
	int size = matrix->height*matrix->width;
	if (matrix == NULL){
		printf("Erro: Matriz não declarada.");
		return 0;
	}
	if(size % 8*number_of_threads != 0){
    	printf("Erro: Tamanho do array tem de ser multiplo de number_of_threads * 8).\n");
		return 0;
	}
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int i=0;i<number_of_threads;i++){
		thread_array[i].size=size;
		thread_array[i].offset=(size / number_of_threads) * i;
		thread_array[i].matrix=matrix;
		thread_array[i].scalar_value=scalar_value;
		if(i==number_of_threads-1)
			thread_array[i].end=1;
		if(i!=number_of_threads-1)
			thread_array[i].end=0;
		thread_created=pthread_create(&threads,NULL,avx_scalar,(void*)&thread_array[i]);
		if(thread_created){
			printf("Erro: pthread_create deu erro");
			exit(-1);
		}
	}
	pthread_attr_destroy(&attr);
	for(int j = 0; j < number_of_threads; j++) {
    	thread_created = pthread_join(threads[j], &status);
    	if(thread_created) {
    		printf("Erro: pthread_join deu erro");
      		exit(-1);
    	}
  	}
	return 1;
}


void * avx_multiply(void * arg) {
	int final;
	int current_row;
	struct multiplication_thread * current;
	__m256 a_vector;
	__m256 b_vector;
	__m256 c_vector;
	__m256 escalar_a_b;
	Matrix *m_a = current->A;
	Matrix *m_b = current->B;
	Matrix *m_c = current->C;
	float * a_next = m_a->rows;
	float * b_next = m_b->rows;
	float * c_next = m_c->rows;
	current = (struct multiplication_thread*) arg;
 	final= (current->size / number_of_threads) + current->offset;
	if(current->end){
    	final = current->size;
	}
/* Mais lento - versao antiga do t2
	for(int i = 0 ; i < m_a->height; i++, a_next+= 8){
		c_next=m_c->rows+(i*m_b->width);

		for(int j = 0;j<m_a->width;j++){
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
	*/
//mais rapido
	for(int i = current->offset; i < final; i++, a_next++){
    	a_vector = _mm256_set1_ps(*a_next);
    	b_next = m_b->rows;
	    current_row = i / m_a->width;
    	c_next = m_c->rows + current_row * m_b->width;
    	for(int j = 0; j < m_b->width; j+=8, b_next+=8, c_next+=8){
      		b_vector = _mm256_load_ps(b_next);
      		c_vector = _mm256_load_ps(c_next);
			escalar_a_b = _mm256_fmadd_ps(a_vector, b_vector, c_vector);
      		_mm256_store_ps(c_next, escalar_a_b);
    	}
	}
	
  pthread_exit(NULL);
}

int matrix_matrix_mult(Matrix * m_a, Matrix * m_b, Matrix * m_c){
	pthread_t threads[number_of_threads];
	pthread_attr_t attr;
	int thread_created;
	struct multiplication_thread thread_array[number_of_threads];
	void *status;
	int size = m_a->height*m_a->width;
	if(m_a == NULL || m_b == NULL || m_c ==NULL){
		printf("Erro: Uma ou mais matrizes não declaradas.\n");
		return 0;
	}
	if(m_a->width != m_b->height){
		printf("Erro: A largura da matriz A precisa ser igual a altura da matriz B.\n");
		return 0;
	}	
	if(size % 8*number_of_threads != 0){
    	printf("Erro: tem de ser multiplo de 8*number_of_threads.\n");
    	return 1;
  	}
  	pthread_attr_init(&attr);
  	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(int i = 0; i < number_of_threads; i++){
    thread_array[i].size = size;
    thread_array[i].offset = (size / number_of_threads) * i;
    thread_array[i].A = m_a;
    thread_array[i].B = m_b;
    thread_array[i].C = m_c;
    if(i == number_of_threads-1)
      thread_array[i].end = 1;
    if(i != number_of_threads-1)
      thread_array[i].end = 0;
    
    thread_created = pthread_create(&threads[i], NULL, avx_multiply, (void *)&thread_array[i]);
    if (thread_created) {
      printf("Erro: pthread_create deu erro");
      exit(-1);
    }
  }

  pthread_attr_destroy(&attr);
  for(int j = 0; j< number_of_threads; j++) {
    thread_created = pthread_join(threads[j], &status);
    if (thread_created) {
      printf("Erro: pthread_join deu erro");
      exit(-1);
    }
  }
  return 1;
}

