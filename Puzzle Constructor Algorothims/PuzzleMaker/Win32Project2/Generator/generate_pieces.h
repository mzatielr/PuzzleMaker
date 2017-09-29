#pragma once

#ifndef GENERATE_PIECES_H
#define GENERATE_PIECES_H

#include <vector>

#include "../Project1/image.hpp"

using namespace std;
#define pb push_back
#define bin 10

Block* block;
void generateImages(IplImage* img, int n, int height, int width);
void assignMemory(int height,int width,int X);
vector<Block> permute(int n);
#endif
