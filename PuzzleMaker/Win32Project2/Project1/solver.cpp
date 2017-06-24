#include <iostream>
#include <time.h>
#include "image.hpp"
#include <utility>
#include <vector>

#include <opencv/highgui.h>
#include <opencv/cv.h>

#include "MST_solver.h"
#include "GA_solver.h"

using namespace std;

#define TIME_LIMIT 15.0

typedef pair<int,int> pii;
typedef pair<pii,int> ppi;


int N,X;
Images* pieces;

void saveResult(vector<Block> &ans,int height,int width,const string output)
{
	CvScalar pix;

    IplImage * final = cvCreateImage(cvSize(width*N,height*N),IPL_DEPTH_8U,3);
    for(int i = 0;i<X;i++)
    {
        int start = i/N;
        int stop = i%N;
        start = start*height;
        stop = stop*width;
        for(int j = 0; j<height; j++)
        {
            for(int k = 0; k<width;k++)
            {
                for(int h = 0; h<3;h++)
                    pix.val[h]=ans[i].image[j][k].val[h];
                cvSet2D(final,j+start,k+stop,pix);
            }
        }
    }
    cvSaveImage(output.c_str(),final);
}

int main(int argc, char *argv[])
{
	pieces = new Images();
	pieces->initializeAll();
	N = pieces->N;
	X = N*N;

	vector<Block> scrambled = pieces->getScrambledImage();
	saveResult(scrambled, pieces->height, pieces->width, "scrambled_image.jpg");

	if (N > 15)
	{
		MST mst(N, pieces);
		vector<Block> ans = mst.get_mst(pieces->height, pieces->width);
		saveResult(ans, pieces->height, pieces->width, "solved_image.jpg");
	}
	else
	{
		//cout << "N lower than 15 " << endl;
		GA ga(N, pieces);
		vector<Block> ans=ga.runAlgo(pieces->height,pieces->width);
		saveResult(ans,pieces->height,pieces->width,"solved_image.jpg");
	}
	return 0;
}

