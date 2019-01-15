  
	#include "SDL2/SDL.h"
	#include "SDL2/SDL_mixer.h"
	#include <unistd.h>
        #include <stdio.h>
       int callinfinityloop();
	//static const char *MY_COOL_MP3 = "music1.mp3";
	int main(int argc, char *argv[]) {
	    int result = 0,i=0;;
	    int flags = MIX_INIT_MP3;
		Mix_VolumeMusic(10);
	    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
		printf("Failed to init SDL\n");
		//exit(1);
	    }
           
	    if (flags != (result = Mix_Init(flags))) {
		printf("Could not initialize mixer (result: %d).\n", result);
		printf("Mix_Init: %s\n", Mix_GetError());
	       // exit(1);
	    }

	    Mix_OpenAudio(22050, AUDIO_S16SYS, 2, 640);
           // Mix_VolumeMusic(128);
	    Mix_Music *music = Mix_LoadMUS(argv[1]);
	    if( !music){
	    printf("%s",Mix_GetError());}
	    Mix_PlayMusic(music, 1);
            callinfinityloop();
            Mix_FreeMusic(music);
	    
	   // printf("volume was    : %d\n", Mix_VolumeMusic(MIX_MAX_VOLUME));
             /*music = Mix_LoadMUS(argv[2]);
              Mix_VolumeMusic(100);
             Mix_PlayMusic(music, 1);
            callinfinityloop(10);
            Mix_FreeMusic(music);*/
	   /*  sleep(15);
               SDL_PauseAudio(0);
               SDL_Delay(2500);
               SDL_CloseAudio();*/
            // Mix_HaltMusic();
            // Mix_PauseMusic();
            // Mix_ResumeMusic();
	      SDL_Quit();
            // SDL_QuitRequested();
	    
	}

       int callinfinityloop()
       {
	for ( ; ;) {
		
                 if(Mix_PlayingMusic ()==0)
		          { 
			    return 0;
			  }
              
	           }
    }
//gcc -o Gplay music.c -lSDL2 -lSDL2_mixer
//./Gplay "music1.mp3"

