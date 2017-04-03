#include "stdio.h"
#include <stdlib.h>
#include <string.h>
//#include <thrust/sort.h>
//#include <thrust/extrema.h>

 void BubbleSort(int source[], int dest[], int weight[], int Nedges)
 {
	int i, j, temp, temp2, temp3;
	for (i = 0; i < (Nedges - 1); ++i)
	{
      for (j = 0; j < Nedges - 1 - i; ++j )
      {
           if (source[j] > source[j+1])
           {
                temp = source[j+1];
				temp2 = dest[j+1];
				temp3 = weight[j+1];
                source[j+1] = source[j];
				dest[j+1] = dest[j];
				weight[j+1] = weight[j];
                source[j] = temp;
				dest[j] = temp2;
				weight[j] = temp3;
           }
      }
	}
 } 
 
 void Weighted(int source[], int dest[], int weight[], int words[], int Nedges)
 {
	 int z = 0;
	for( z = 0; z < Nedges; z++)
	{
		source[z] = words[z*3];
		//printf("source[%d] = %d\n", z, source[z]);
		dest[z] = words[(z*3)+1];
		weight[z] = words[(z*3)+2];
	}
	 
 }
 
 void UnWeighted(int source[], int dest[], int weight[], int words[], int Nedges)
 {
	 int z = 0;
	for( z = 0; z < Nedges; z++)
	{
		source[z] = words[z*2];
		//printf("source[%d] = %d\n", z, source[z]);
		dest[z] = words[(z*2)+1];
		weight[z] = 1;
	}
	 
 }
 
 // ex:- ./sorter input.txt weighted output.txt dest
 // or
 // ./sorter input2.txt unweighted output2.txt source 
 
int main (int argc, char* argv [])
{
	
	char* inputname = argv[1]; // input file name
	char* weightornot = argv[2];// weighted or unweighted
	char* output = argv[3]; // output file name
	char* sourcedest = argv[4]; // sorted by source or dest
	
	FILE *fp = fopen(inputname, "r");
	int Nedges = 0;
	char input[512];
	while (fgets( input, 512, fp))
	{
		Nedges++;
		//printf("Line:%d -> %s", line, input);
	}
	rewind(fp);
	
	int source[Nedges] ;
	int dest[Nedges];
	int weight[Nedges];

	int x = 0;
	int var;
	int words[Nedges*3];
	char* token;
	
	while ( fgets(input, 512, fp ) != NULL) // read a line at a time.
    {
         token = strtok (input," 	\n");
		 while (token != NULL)
         {
			 sscanf(token, "%d", &var);
			 words[x] = var;
			//printf("words[%d] =  %d\n", x, words[x]);
            token = strtok(NULL, " 	\n");
			x++;
         }
    }
	
	fclose(fp);
	
	if ( strcmp(weightornot , "weighted") ){
    	UnWeighted( source, dest, weight, words, Nedges);		
	
	}else{
          Weighted( source, dest, weight, words, Nedges);
        }  

		
		
	if ( strcmp(sourcedest, "source") ){
    	
		BubbleSort(dest, source, weight, Nedges); 
	}else{
         BubbleSort(source, dest, weight, Nedges); 	
         } 
	// Will sort with respect to dest in this case. To sort source, just swap dest and source. Works for both weighted and unweighted.
	 
	
	
	FILE *fp2 = fopen(output, "w");
	int c;
	for ( c = 0 ; c < Nedges ; c++ )
	{	
		fprintf(fp2, "%ld	", source[c]);
		fprintf(fp2, "%ld	", dest[c]);
		fprintf(fp2, "%ld\n", weight[c]);		
	}

	fclose(fp2);
	
	return 0;
}