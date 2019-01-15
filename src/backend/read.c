/*#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>

char** str_split( char* str, char delim, int* numSplits )
{
    char** ret;
    int retLen;
    char* c;

    if ( ( str == NULL ) ||
        ( delim == '\0' ) )
    {
        
        ret = NULL;
        retLen = -1;
    }
    else
    {
        retLen = 0;
        c = str;

       
        do
        {
            if ( *c == delim )
            {
                retLen++;
            }

            c++;
        } while ( *c != '\0' );

        ret = malloc( ( retLen + 1 ) * sizeof( *ret ) );
        ret[retLen] = NULL;

        c = str;
        retLen = 1;
        ret[0] = str;

        do
        {
            if ( *c == delim )
            {
                ret[retLen++] = &c[1];
                *c = '\0';
            }

            c++;
        } while ( *c != '\0' );
    }

    if ( numSplits != NULL )
    {
        *numSplits = retLen;
    }
 return ret;
}




typedef struct
{
	char  detection_sensitivity[6];	// sensitivity
	char  alert_sensitivity[6];
	char  frame_sensitivity[6];
} sensitivity;

typedef struct
{
	sensitivity sens_object ;		// basic
	char debug[5];
} basic;

typedef struct
{
	char  EAR[5];				// advance
	char  MAR[5];
	char  FAR[5];
	char  DROWSY_CONSEC_FRAMES[4];
	char  YAWN_CONSEC_FRAMES[4];
	char  INATT_CONSEC_FRAMES[4];
} advanced;

typedef struct
{
	char type[11];
	basic basic_object;			// parameters
	advanced adv_object;
} parameters;


 parameters sending ;
 

int main ()
{    
     char* strCpy;
    char** split;
    char c[52],ch;
     int i=0,num;
    // Open person.dat for reading
   FILE *infile = fopen ("person.dat", "r");
    if (infile == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit (1);
    }
    while((ch = getc(infile)) != EOF)
    {
      c[i]=ch;
      i++;
      }
  printf("%s\n",c);
  strCpy = malloc( strlen( c ) * sizeof( *strCpy ) );
   strcpy( strCpy, c);
				
   split = str_split( strCpy, ',', &num );
   printf("%d\n",num-1);
        strcpy(sending.type, split[0]); 
        strcpy(sending.adv_object.EAR ,split[5]);
        strcpy(sending.adv_object.MAR ,split[6]);
        strcpy(sending.adv_object.FAR , split[7]);
        strcpy(sending.adv_object.DROWSY_CONSEC_FRAMES ,split[8]);
        strcpy(sending.adv_object.YAWN_CONSEC_FRAMES ,split[9]);
        strcpy(sending.adv_object.INATT_CONSEC_FRAMES , split[10]);
        strcpy(sending.basic_object.sens_object.detection_sensitivity , split[1]);
        strcpy(sending.basic_object.sens_object.alert_sensitivity, split[2]);
        strcpy(sending.basic_object.sens_object.frame_sensitivity , split[3]);
        strcpy(sending.basic_object.debug , split[4]);
        puts(sending.type);
        puts(sending.adv_object.EAR);
        puts(sending.adv_object.MAR);
        puts(sending.adv_object.FAR);
        puts(sending.adv_object.DROWSY_CONSEC_FRAMES);
        puts(sending.adv_object.YAWN_CONSEC_FRAMES);
        puts(sending.adv_object.INATT_CONSEC_FRAMES);
        puts(sending.basic_object.sens_object.detection_sensitivity);
        puts(sending.basic_object.sens_object.alert_sensitivity);
        puts(sending.basic_object.sens_object.frame_sensitivity);
        puts(sending.basic_object.debug);
       
    return 0;
}*/
#include <stdio.h>
#include <stdlib.h>
 
int main()
{
   
   FILE *fp;
 
   char ch,c[52];
   fp = fopen("person.dat","r"); // read mode
    int i=0;
   if( fp == NULL )
   {
      perror("Error while opening the file.\n");
      exit(EXIT_FAILURE);
   }
 
   
 fgets(c,52,fp);
   
  printf("%s",c);
   fclose(fp);
   return 0;
}

