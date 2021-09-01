#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct matrix {
unsigned long int height; //número de linhas da matriz (múltiplo de 8)
unsigned long int width; //número de colunas da matriz (múltiplo de 8)
float *rows; //sequência de linhas da matriz (height*width elementos)
} Matrix;



void showMatrix(Matrix matrix){

	unsigned long int mH = matrix.height;
	unsigned long int mW = matrix.width;

	printf("[ ");
	for(int i=0;i<mH; i++){
		for(int j=0; j<mW; j++){
			printf(" %.1f ",matrix.rows[i*mW + j]);
		}
		printf("\n");
	}
	printf("]\n");


}


void fillMatrixWithFile(Matrix matrix, char* filename, int h, int w){
	FILE *file;
	file = fopen (filename, "rb"); 
    if (file == NULL) { 
        fprintf(stderr, " - Erro: problema ao abrir arquivo\n"); 
        exit (1); 
    }

    for(int i=0;i<h; i++){
		for(int j=0; j<w; j++){
			fread (&matrix.rows[i*matrix.width + j], sizeof(float), 1, file);
		}
	}
	fclose(file);
}



void newMatrix(char * filename,float num,int height,int width){
	FILE *file;
	file = fopen (filename, "wb");
	if (file == NULL) { 
        fprintf(stderr, "- Erro: problema ao abrir arquivo \n"); 
        exit (1); 
    }

	for(int i=0;i<height; i++){
		for(int j=0;j<width; j++){
			fwrite (&num, sizeof(float), 1, file);
		}
	}
	
   	fclose(file);

}



int main (int argc, char *argv[]){

	char* filename = argv[1];
	unsigned long int width = atoi(argv[2]);
	unsigned long int height = atoi(argv[3]);
	unsigned long int value = atoi(argv[4]);

	Matrix m;


	newMatrix(filename,value,height,width);


   	m.height=width;
	m.width=height;
	m.rows=(float*)malloc(m.height*m.width*sizeof(float));
	fillMatrixWithFile(m,filename,height,width);
	printf("\n Matrix \n");
	showMatrix(m);
	
   	return 0;
}