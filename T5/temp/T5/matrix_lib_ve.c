#include <stdint.h>
#include <stdio.h>
#include <veo_hmem.h>

uint64_t scalar_mult(unsigned long matrix_size, float * ve_identifier, float scalar, int num_of_threads) {
    int tid=0;
	int	threads = 0;
    ve_identifier = (float *) veo_get_hmem_addr(ve_identifier);
    fflush(stdout);
    omp_set_num_threads(num_threads); 
    #pragma omp parallel private(threads, tid)
    {
        tid = omp_get_thread_num();
        threads = omp_get_num_threads();
        if (tid == 0) {
           // printf("ve_add : number of threads = %d\n", nthreads);
            fflush(stdout);
        }
        unsigned long i;
		unsigned long items_per_thread;
		unsigned long items_last_thread;
		unsigned long first; 
		unsigned long roof;
        items_per_thread = matrix_size / threads;
        items_last_thread = matrix_size % threads;
        first = tid * items_per_thread;
        roof = first+ items_per_thread;
        if (tid == threads - 1) {
			roof += items_last_thread;
		}
        for (i = first; i < roof; ++i) {
            ve_identifier[i] *= scalar;
        }
    }
    return 0;
}

uint64_t matrix_mult(unsigned long matrix_size, float * ve_identifier, float scalar, int num_of_threads) {
    omp_set_num_threads(num_of_threads); 
    return 0;
}
