#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define DEBUG_MODE_OFF (0)
#define DEBUG_MODE_ON (1)
#define BACKGROND_COLOR_CHANNELS (3)
#define BACKGROND_COLOR_RED (0)
#define BACKGROND_COLOR_GREEN (1)
#define BACKGROND_COLOR_BLUE (2)

#define RESULT_FILE_WORDS (200)
#define MAX_CLUSTERS (4) /* number of cluster */
#define DEBUG_MODE (0)/*If u input 0,this program runs as runnning mode. 1 is debug mode*/

/*function DebugClusterNumber*/
int DebugClusterNumber(int Number){
	if (Number > MAX_CLUSTERS)
	{
		return 1;
	}
	else
	{
		return Number;
	}
}

int main(int argc, char** argv)
{
	int i, j, size;
	IplImage *src_img = 0, *dst_img = 0;
	CvMat tmp_header;
	CvMat *clusters, *points, *tmp;
	CvMat *count = cvCreateMat(MAX_CLUSTERS, 1, CV_32SC1);
	CvMat *centers = cvCreateMat(MAX_CLUSTERS, 3, CV_32FC1);
	const char *imagename;
	//set background color
	int background_color[BACKGROND_COLOR_CHANNELS] = { 255, 135, 60 };

	//set place u save image file.
	char file_name[] = "";
	char file_extension[] = ".bmp";

	// (1)load a specified file as a 3-channel color image
	imagename = argc > 1 ? argv[1] : "";

	printf("got path: %s", imagename);
	src_img = cvLoadImage(imagename, CV_LOAD_IMAGE_COLOR);
	if (src_img == 0)
		return -1;

	size = src_img->width * src_img->height;
	dst_img = cvCloneImage(src_img);
	clusters = cvCreateMat(size, 1, CV_32SC1);
	points = cvCreateMat(size, 1, CV_32FC3);

	// (2)reshape the image to be a 1 column matrix
#if 0
	tmp = cvCreateMat(size, 1, CV_8UC3);
	tmp = cvReshape(src_img, &tmp_header, 0, size);
	cvConvert(tmp, points);
	cvReleaseMat(&tmp);
#else
	for (i = 0; i<size; i++) {
		points->data.fl[i * 3 + 0] = (uchar)src_img->imageData[i * 3 + 0];
		points->data.fl[i * 3 + 1] = (uchar)src_img->imageData[i * 3 + 1];
		points->data.fl[i * 3 + 2] = (uchar)src_img->imageData[i * 3 + 2];
	}
#endif

	// (3)run k-means clustering algorithm to segment pixels in RGB color space
	cvKMeans2(points, MAX_CLUSTERS, clusters,
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		1, 0, 0, centers, 0);

	// (4)make a each centroid represent all pixels in the cluster
	//If u selected DEBUG_MODE_OFF,All clusters output.
	if (DEBUG_MODE == DEBUG_MODE_OFF){
		for (i = 0; i<size; i++){
			int idx = clusters->data.i[i];
			dst_img->imageData[i * 3 + 0] = (char)centers->data.fl[idx * 3 + 0];
			dst_img->imageData[i * 3 + 1] = (char)centers->data.fl[idx * 3 + 1];
			dst_img->imageData[i * 3 + 2] = (char)centers->data.fl[idx * 3 + 2];
		}
		// (5)show source and destination image, and quit when any key pressed
		cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
		cvShowImage("src", src_img);
		cvNamedWindow("low-color", CV_WINDOW_AUTOSIZE);
		cvShowImage("low-color", dst_img);
		cvWaitKey(0);
		cvDestroyWindow("src");
		cvDestroyWindow("low-color");
	}
	//If u selected DEBUG_MODE_ON, cluster which u selected outputs.
	//cluster which u didn't select is painted black color.
	else if (DEBUG_MODE == DEBUG_MODE_ON){
		for (j = 0; j<MAX_CLUSTERS; j++)
		{
			//conbine strings
			char file[RESULT_FILE_WORDS] = "";//DON'T DELETE THIS! OR VARIABLE file ISN'T INITIALIZE.
			//generate saved file name.
			sprintf(file, "%s%d%s", file_name, j, file_extension);
			for (i = 0; i<size; i++)
			{
				int idx = clusters->data.i[i];
				if (j == idx)
				{
					dst_img->imageData[i * 3 + 0] = (char)centers->data.fl[idx * 3 + 0];
					dst_img->imageData[i * 3 + 1] = (char)centers->data.fl[idx * 3 + 1];
					dst_img->imageData[i * 3 + 2] = (char)centers->data.fl[idx * 3 + 2];
				}
				else
				{
					dst_img->imageData[i * 3 + 0] = background_color[BACKGROND_COLOR_BLUE];
					dst_img->imageData[i * 3 + 1] = background_color[BACKGROND_COLOR_GREEN];
					dst_img->imageData[i * 3 + 2] = background_color[BACKGROND_COLOR_RED];
				}
			}
			cvSaveImage(file, dst_img, 0);
			printf("cluster %d image save completed.\n", j);
		}
	}
	cvReleaseImage(&src_img);
	cvReleaseImage(&dst_img);
	cvReleaseMat(&clusters);
	cvReleaseMat(&points);
	cvReleaseMat(&count);
	return 0;
}