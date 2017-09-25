//
//  main.cpp
//  PuzzleSolver
//
//  Created by Joe Zeimen on 4/4/13.
//  Copyright (c) 2013 Joe Zeimen. All rights reserved.
//

#include <iostream>
#include <string.h>
#include "puzzle.h"
#include <cassert>
#include "utils.h"
#include "PuzzleDisjointSet.h"
#include <Windows.h>
#include <stdint.h>

//Dont forget final "/" in directory name.
static const std::string input = "C:\\jig\\Scans\\";
static const std::string output = "C:\\jig\\finaloutput.png";
/*
// MSVC defines this in winsock2.h!?
typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
*/
int main(int argc, const char * argv[])
{
    
	
    std::cout << "Starting..." << std::endl;
    //timeval time;
    ///gettimeofday(&time, NULL);
   // long millis = (time.tv_sec * 1000) + (time.tv_usec / 1000);
    //long inbetween_millis = millis;
     //Toy Story Color & breaks with median filter, needs filter() 48 pc
     //puzzle puzzle(input+"Toy Story\\", 200, 22, false);

    
    //Toy Story back works w/ median filter 48pc
   // puzzle puzzle(input+"Toy Story back/", 200, 50);
    
    //Angry Birds color works with median, or filter 24 pc
    //puzzle puzzle(input+"Angry Birds/color/",300,30);

    //Angry Birds back works with median 24 pc
//    puzzle puzzle(input+"Angry Birds/Scanner Open/",300,30);
    
      //Horses back not numbered 104 pc
//    puzzle puzzle(input+"horses/", 380, 50);

    //Horses back numbered 104 pc
    puzzle puzzle(input+"horses numbered/", 380, 50);

    //gettimeofday(&time, NULL);
    //std::cout << std::endl << "time to initialize:"  << (((time.tv_sec * 1000) + (time.tv_usec / 1000))-inbetween_millis)/1000.0 << std::endl;
  //inbetween_millis = ((time.tv_sec * 1000) + (time.tv_usec / 1000));
    
    puzzle.solve();
   // gettimeofday(&time, NULL);
    //std::cout << std::endl << "time to solve:"  << (((time.tv_sec * 1000) + (time.tv_usec / 1000))-inbetween_millis)/1000.0 << std::endl;
  //inbetween_millis = ((time.tv_sec * 1000) + (time.tv_usec / 1000));
    puzzle.save_image(output);
    //gettimeofday(&time, NULL);
    //std::cout << std::endl << "Time to draw:"  << (((time.tv_sec * 1000) + (time.tv_usec / 1000))-inbetween_millis)/1000.0 << std::endl;
    
    
    //gettimeofday(&time, NULL);
    //std::cout << std::endl << "total time:"  << (((time.tv_sec * 1000) + (time.tv_usec / 1000))-millis)/1000.0 << std::endl;

   // system("/usr/bin/open /tmp/final/finaloutput.png");
    
    return 0;
}


