#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <pthread.h>
#include <errno.h>
#include <arpa/inet.h>
# define D_PORT 8081
#include "SDL2/SDL.h"
#include "SDL2/SDL_mixer.h"


int sound();
typedef struct {

    float speedfloat;          // structure use for storing gps data 
    int date;
    int time;
       } gpsinf;
 
gpsinf gps_want_data;

char** str_split( char* str, char delim, int* numSplits )
{
    char** ret;
    int retLen;
    char* c;

    if ( ( str == NULL ) ||
        ( delim == '\0' ) )
    {
        /* Either of those will cause problems */
        ret = NULL;
        retLen = -1;
    }
    else
    {
        retLen = 0;
        c = str;

        /* Pre-calculate number of elements */
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


void *myThreadFun()
{
  int serial_port; 
  char dat,buff[100],GGA_code[6];
  unsigned char IsitGGAstring=0;
  char Parsed_Frame[30];
  char line[80];
  unsigned char GGA_index=0;
  unsigned char is_GGA_received_completely = 0;
    char* strCpy;
    char** split;
    int num;
    int i;

  if ((serial_port = serialOpen ("/dev/ttyS0", 9600)) < 0)		
  {
    fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno)) ;
    
  }

  if (wiringPiSetup () == -1)							
  {
    fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno)) ;
   
  }

  while(1){
	  
		 if(serialDataAvail (serial_port) )			
		  { 
			  dat = serialGetchar(serial_port);
                   				
			  if(dat == '$')
                        {
				
				IsitGGAstring = 1;
				GGA_index = 0;
			 }
			  if(IsitGGAstring ==1)
			 {
				line[GGA_index++] = dat;
				
				if(dat=='\r')
				 {
				 if(strstr(line, "$GPRMC"))
				{
				 printf("%s\n",line);
                                 strCpy = malloc( strlen( line ) * sizeof( *strCpy ) );
				    strcpy( strCpy, line);
				
				    split = str_split( strCpy, ',', &num );
				
				    if ( split == NULL )
				    {
				        puts( "str_split returned NULL" );
				    }
				    else
				    {
				        printf( "%i Results: \n", num );
				
				        for ( i = 0; i < num; i++ )
				        {   
				            //puts( split[i] );
					    switch(i+1)
						 {
						 case 1:printf("%s\n",split[i]);
							break;
						 case 2:printf("Time:%s\n",split[i]);
							break;
			                         case 3:printf("Navigation:%s\n",split[i]);
							break;
						 case 4:printf("Latitude:%s\n",split[i]);
							break;
						 case 5:printf("North:%s\n",split[i]);
							break;
						 case 6:printf("Longitude:%s\n",split[i]);
							break;
						 case 7:printf("West:%s\n",split[i]);
							break;
						 case 8:gps_want_data.speedfloat=(atof(split[i])*1.852);
							printf("Speed:%f\n",gps_want_data.speedfloat);
							break;
			                         case 9:printf("Course:%s\n",split[i]);
							break;
						 case 10:printf("Date:%s\n",split[i]);
							break;
						 case 11:printf("Magnetic:%s\n",split[i]);
							break;
						 case 12:printf("East:%s\n",split[i]);
							break;
						 case 13:printf("Mandatory checksum:%s\n",split[i]);
							break;
				 		 }
				        }
   				    }
        			}
			  }
		        }
		
		  }
		
	}
	
}



typedef struct
{  
	char  detection_sensitivity[5];	    // sensitivity
	char  alert_sensitivity[4];
	char  frame_sensitivity[4];
} sensitivity;

typedef struct
{
	sensitivity sens_object ;		// basic
	char debug[5];
} basic;

typedef struct
{
	char  EAR[4];				// advance
	char  MAR[4];
	char  FAR[4];
	char  DROWSY_CONSEC_FRAMES[3];
	char  YAWN_CONSEC_FRAMES[3];
	char  INATT_CONSEC_FRAMES[3];
} advanced;

typedef struct
{
	char type[11];
	basic basic_object;			// parameters
	advanced adv_object;
} parameters;

parameters sending;



  int main ()
{
    int i=0;
    pthread_t tid;
    
    pthread_create(&tid, NULL, myThreadFun, NULL);
    //pthread_create(&tid1,NULL,thread1,NULL);
   
  strcpy(sending.type,"parameters\n");                                      // sample data
strcpy(sending.adv_object.EAR,"0.4\n");                                   // sample data
strcpy(sending.adv_object.MAR,"0.4\n");                                   // sample data
strcpy(sending.adv_object.FAR,"0.4\n");                                   // sample data
strcpy(sending.adv_object.DROWSY_CONSEC_FRAMES,"15\n");                   // sample data
strcpy(sending.adv_object.YAWN_CONSEC_FRAMES,"15\n");                     // sample data
strcpy(sending.adv_object.INATT_CONSEC_FRAMES,"15\n");                    // sample data
strcpy(sending.basic_object.sens_object.detection_sensitivity,"HIGH\n");  // sample data
strcpy(sending.basic_object.sens_object.alert_sensitivity,"LOW\n");       // sample data
strcpy(sending.basic_object.sens_object.frame_sensitivity,"MED\n");       // sample data
strcpy(sending.basic_object.debug,"True\n");                              // sample data

 char message[14],message1[5],message2[3],message3[4],message4[4],test[7];
 int status;
 char client_reply[300];
printf("Options: \n");
 printf("1. Send parameters\n");
 printf("2. Test face detection\n");
 printf("3. Option to train\n");
 printf("4. Start application\n");
 printf("5. Stop application\n");

 int sock, choice;

 struct sockaddr_in source, destination;
while(1)
 {
 memset(&destination, 0, sizeof(destination));
 destination.sin_port = htons(D_PORT);
 destination.sin_addr.s_addr = inet_addr("127.0.0.1");   // destination port
 destination.sin_family = AF_INET;
 sock = socket(AF_INET, SOCK_STREAM, 0);
 if(socket(AF_INET, SOCK_STREAM, 0) < 0)
	printf("Socket not created.\n");
if(connect(sock, (struct sockaddr *)&destination ,sizeof(destination)) < 0)
	printf("Connection failed.\n");


 while(1)
 {
 sleep(1);
  printf("Enter option: ");
  //scanf("%d", &choice);
  sound();
  switch(choice)
  {
    case 1:if(status==4)
	    {
		//strcpy(message,"stop");
		//send(sock,(char *)&message , sizeof(message), 0);
		printf("please stop application first \n");
                 break;
	     }
	    else
	     {	
		send(sock, (char *)&sending , sizeof(sending), 0);
                break;
	     }
	    

    case 2: strcpy(message,"face_detection");			// test presence of face, "face_detection"
	    send(sock,(char *)&message, sizeof(message), 0);
	    break;

    case 3: strcpy(message1,"train");                            // Option to train, "train"
	    send(sock,(char *)&message1 , sizeof(message1), 0);
	    break;

    case 4: strcpy(message2,"run"); 				// Start application, "run"
 	    send(sock, (char *)&message2 , sizeof(message2), 0);
	    status=4;
	    break;

    case 5: strcpy(message3,"exit");				// Stop application, "exit"
	    send(sock,(char *)&message3 , sizeof(message3), 0);
	    break;

    case 6: status=5;
	    strcpy(message4,"stop");				// Stop application, "stop"
	    send(sock,(char *)&message4 , sizeof(message4), 0);
	    break;

    default: printf("Incorrect option, try again!\n");
	     break;

  }
 
            if(choice == 4)
		{
		 if(recv(sock , client_reply , 300, 0)<0);
		 printf("Not receive");
                 printf("%s",client_reply);                 // receive data from server after "run" command
		 
		}
 

 }

 close(sock);
 
}
   pthread_join(tid, NULL);
   
	return 0;
}     



int sound() {
	    int result = 0,i=0;;
	    int flags = MIX_INIT_MP3;
            static const char *MY_COOL_MP3 = "Sleep.wav";
	    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
		printf("Failed to init SDL\n");
		//exit(1);
	    }
            // printf("hemant");
	    if (flags != (result = Mix_Init(flags))) {
		printf("Could not initialize mixer (result: %d).\n", result);
		printf("Mix_Init: %s\n", Mix_GetError());
	       // exit(1);
	    }

	    Mix_OpenAudio(22050, AUDIO_S16SYS, 2, 640);
            
	    Mix_Music *music = Mix_LoadMUS(MY_COOL_MP3);
	    if( !music){
	    printf("%s",Mix_GetError());}
	    Mix_PlayMusic(music, 1);
            callinfinityloop();
            Mix_FreeMusic(music);
	   /*  sleep(15);
               SDL_PauseAudio(0);
               SDL_Delay(2500);
               SDL_CloseAudio();*/
            // Mix_HaltMusic();
            // Mix_PauseMusic();
            // Mix_ResumeMusic();
	     //SDL_Quit();
            // SDL_QuitRequested();
	    return 0;
	}

       int callinfinityloop()
       {
	for ( ; ;) {
		//SDL_Delay(250);
                 
 		 //break ;
		//Mix_HaltMusic();
		 //Mix_PauseMusic();
                 //sleep(12);
		//Mix_ResumeMusic();
		if(Mix_PlayingMusic()==0)
		{
		return 0;
		}
	    }
		} 
//gcc -o gps gpsinf1.c -lwiringPi -lwiringPiDev -lpthread
	