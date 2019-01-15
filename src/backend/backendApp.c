#include "SDL2/SDL.h"
#include "SDL2/SDL_mixer.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>
# define D_PORT 8081
void *connection_handler(void *);
int callinfinityloop();
int sound(char *);
int client;
 typedef struct {

    float speedfloat;          // structure use for storing gps data 
    int date;
    int time;
       } gpsinf;
 
gpsinf gps_want_data;

char** str_split1( char* str, char delim, int* numSplits )
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
  serialPuts (serial_port, "$PMTK314,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*29\r\n") ;
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
				// printf("%s\n",line);
                                 
                                 strCpy = malloc( strlen( line ) * sizeof( *strCpy ) );
				    strcpy( strCpy, line);
				
				    split = str_split1( strCpy, ',', &num );
				
				    if ( split == NULL )
				    {
				        puts( "str_split returned NULL" );
				    }
				    else
				    {
				      //  printf( "%i Results: \n", num );
				
				        for ( i = 0; i < num; i++ )
				        {   
				            //puts( split[i] );
					  /*  switch(i+1)
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
				 		 }*/
				        }
   				    }
        			}
			  }
		        }
		
		  }
		
	}
	
}


   
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
				char  detection_sensitivity[6];	    // sensitivity
				char  alert_sensitivity[5];
				char  frame_sensitivity[5];
			} sensitivity;
			
			typedef struct
			{
				sensitivity sens_object ;		// basic
				char debug[6];
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
				char type[12];
				basic basic_object;			// parameters
				advanced adv_object;
			} parameters;
			
			parameters sending;




int main (void)
{

    char* strCpy;
    char** split;
    char c[52],ch,bluetext[220];
     int i=0,num;
    char message[14],message1[5],message20[4],message21[4],message3[4],message4[4],test[7];
    int status; 

   char buf[1024] = {0};	
   int bluetoothSocket, bytes_read;		 
   struct sockaddr_rc loc_addr = {0};
   struct sockaddr_rc client_addr = {0};

   socklen_t opt = sizeof(client_addr);	

    // Open person.dat for reading
   FILE *infile = fopen ("person.dat", "r");
	    if (infile == NULL)
	    {
	        fprintf(stderr, "\nError opening file\n");
	        exit (1);
	    }
         fgets(c,52,infile);
         printf("%s\n",c);
         strCpy = malloc( strlen( c ) * sizeof( *strCpy ) );
         strcpy( strCpy, c);
				
         split = str_split( strCpy, ',', &num );
	 /*for(i=0;i<num;i++)
           {
             strcat(split[i],"*");
		}*/

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

         strcat(sending.type,"*");
         strcat(sending.adv_object.EAR,"*");
         strcat(sending.adv_object.MAR,"*");
         strcat(sending.adv_object.FAR,"*");
         strcat(sending.adv_object.DROWSY_CONSEC_FRAMES,"*");
         strcat(sending.adv_object.YAWN_CONSEC_FRAMES,"*");
         strcat(sending.adv_object.INATT_CONSEC_FRAMES,"*");
         strcat(sending.basic_object.sens_object.detection_sensitivity,"*");
         strcat(sending.basic_object.sens_object.alert_sensitivity,"*");
         strcat(sending.basic_object.sens_object.frame_sensitivity,"*");
         strcat(sending.basic_object.debug,"*");                             // sample data

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
         puts(sending.basic_object.debug);                             // sample data

 	
         
	 /*printf("Options: \n");
	 printf("1. Send parameters\n");
	 printf("2. Test face detection\n");
	 printf("3. Option to train\n");
	 printf("4. Start application\n");
	 printf("5. Exit application\n");
         printf("6. Stop application\n");*/
	 pthread_t thread_id,tid;
         pthread_create(&tid, NULL, myThreadFun, NULL);
	 int sock, choice,nbytes;
	 choice = 4;
	 struct sockaddr_in source, destination;
	
	 memset(&destination, 0, sizeof(destination));
	 destination.sin_port = htons(D_PORT);
	 destination.sin_addr.s_addr = inet_addr("127.0.0.1");   // destination port
	 destination.sin_family = AF_INET;
         bdaddr_t my_bdaddr_any = {0, 0, 0, 0, 0, 0};	
 
	bluetoothSocket = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);	 
	 
	loc_addr.rc_family = AF_BLUETOOTH;
	loc_addr.rc_bdaddr = (bdaddr_t)my_bdaddr_any;		 		
	loc_addr.rc_channel = (uint8_t) 1;
	
	
	


     while(1)
      {  
         sock = socket(AF_INET, SOCK_STREAM, 0);

	 if(socket(AF_INET, SOCK_STREAM, 0) < 0)
		printf("Socket not created.\n");
        l1:if(connect(sock, (struct sockaddr *)&destination ,sizeof(destination)) < 0)
	       goto l1;
	
	  if( pthread_create( &thread_id , NULL ,  connection_handler , (void*) &sock) < 0)
	       {
		    perror("could not create thread");
	            return 1;
	        }

	 if (bind(bluetoothSocket, (struct sockaddr *)&loc_addr, sizeof(loc_addr)) == -1)	 
	   {
	
	      printf("not bind");
	  
	    }
	 	
	
	
	if( listen(bluetoothSocket, 1) == -1)
	{
	   printf("not listen");
	  
	}
	 printf("Waiting for new client to connect.\n");
		
	  client = accept(bluetoothSocket, (struct sockaddr *)&client_addr, &opt);
		
	  if (client == -1)
	  {
	    printf("Bluetooth connect failed.\n");
	  }
				
	  //printf("Bluetooth connection made.\n");
     sprintf(bluetext,"%s\r\n%s\r\n%s\r\n%s\r\n%s\r\n%s\r\n%s","1. Send parameters","2. Test face detection","3. Option to train","4. Start application (low sensitivity)","5. Start application (high sensitivity)","6. Exit application","7. Stop application","Enter command :");
	  write(client,bluetext,220);
          
	 while(1)
	 {
	   sleep(1);
            ba2str(&loc_addr.rc_bdaddr, buf);
	  fprintf(stderr, "accepted connection from %s\n", buf);
	  memset(buf, 0, sizeof(buf));
	
	  // read data from the client
	  bytes_read = read(client, buf, sizeof(buf));
	  if (bytes_read > 0) 
	  {
	  // printf("Bluetooth bytes received [%s]\n", buf);
           choice = atoi(buf);
	  }
	 
       	
	    // printf("Enter option: ");
	    // printf("%d", choice);
		
  	  
		 switch(choice)
		  {
		    case 1: if(status==4)
			    {
				write(client,"please stop application first \r\n",30);
		                 break;
			     }
			    else
			     {	
                              send(sock, (char *)&sending , sizeof(sending), 0);
		              write(client,"1\r\n",1);
		              break;
			     }
			    
		
		    case 2: strcpy(message,"face_detection");			// test presence of face, "face_detection"
			    send(sock,(char *)&message, sizeof(message), 0);
                            write(client,"2\r\n",1);
			    break;
		
		    case 3: strcpy(message1,"train");                            // Option to train, "train"
			    send(sock,(char *)&message1 , sizeof(message1), 0);
			    write(client,"3\r\n",1);
			    break;
		
		    case 4: if(status == 4)
                             break;
                            strcpy(message20,"run0"); 				// Start application, "run0"
		 	    send(sock, (char *)&message20 , sizeof(message20), 0);
                            write(client,"4\r\n",1);
			    status=4;
		            break;

		   case 5: if(status == 4)
                             break;
                            strcpy(message21,"run1"); 				// Start application, "run1"
		 	    send(sock, (char *)&message21 , sizeof(message21), 0);
                            write(client,"5\r\n",1);
			    status=4;
		            break;
		
		    case 6: strcpy(message3,"exit");				// Stop application, "exit"
			    send(sock,(char *)&message3 , sizeof(message3), 0);
                            write(client,"6\r\n",1);
			    break;
		
		    case 7: status=5;
			    strcpy(message4,"stop");				// Stop application, "stop"
			    send(sock,(char *)&message4 , sizeof(message4), 0);
                            write(client,"7\r\n",1);
			    break;
		
		    default: printf("Incorrect option, try again!\n");
			     break;
		
	        }
	 
        }
  }
 pthread_join(tid, NULL);
 close(sock);
 
 return 0;

}

	void *connection_handler(void *socket_desc)
	{
	    
	    int sock = *(int*)socket_desc;
	    int read_size;
	    char client_reply[50],*cpy,gpsspeed[50],gpsdata[10];
	     FILE *fpt;
           while(1)
            {
	    if( (read_size = recv(sock , client_reply , 50 , 0)) > 0 )
	    {
             fpt = fopen("gps.data","a");
             fprintf(fpt,"%f\n",gps_want_data.speedfloat);
            
              
	    // printf("%s\n",client_reply);

              if(gps_want_data.speedfloat < 5.0)
              {
		 if(!strncmp(client_reply,"333",3))
                   {
	           sound("inattention.wav");
                   fprintf(fpt,"%s\n","Inattention alerts");
                   }
                if(!strncmp(client_reply,"222",3))
                   {
	           sound("yawn.wav");
                   fprintf(fpt,"%s\n","Yawning alert");
                   }
                if(!strncmp(client_reply,"111",3))
                   {
	           sound("Sleep.wav");
                   fprintf(fpt,"%s\n","Sleep alert");
                   }
                  
               }
	     fclose(fpt); 
             sprintf(gpsspeed,"%s\r\n",client_reply);
             write(client,gpsspeed,50);
	    }
	
          sprintf(gpsdata,"%f\r\n",gps_want_data.speedfloat);
          write(client,gpsdata,10);
	  }
	}



	int sound( char *MY_COOL_MP3 ) 
 
	     {
		    int result = 0,i=0;;
		    int flags = MIX_INIT_MP3;
	         
		    if (SDL_Init(SDL_INIT_AUDIO) < 0) 
                     {
			printf("Failed to init SDL\n");
		     }
	            
		    if (flags != (result = Mix_Init(flags))) {
			printf("Could not initialize mixer (result: %d).\n", result);
			printf("Mix_Init: %s\n", Mix_GetError());
		      
		    }
	
		    Mix_OpenAudio(22050, AUDIO_S16SYS, 2, 640);
	            
		    Mix_Music *music = Mix_LoadMUS(MY_COOL_MP3);
		    if( !music){
		    printf("%s",Mix_GetError());}
		    Mix_PlayMusic(music, 1);
	            callinfinityloop();
	            Mix_FreeMusic(music);
		   
		    return 0;
	    }

       int callinfinityloop()
  
	       {
		for ( ; ;)
                   {
			
			if(Mix_PlayingMusic()==0)
			{
			return 0;
			}
		    }
		} 


// gcc -g backendApp.c -o socket -lpthread -lSDL2 -lSDL2_mixer -lwiringPi -lwiringPiDev -lbluetooth

