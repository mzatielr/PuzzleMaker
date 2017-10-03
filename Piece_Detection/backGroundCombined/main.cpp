

//I COMBINED KMEANS AND BACKGROUND REMOVAL ALGORITHM THAT I FOUND, CHECK BELOW FOR NOTES


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv\cv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>   // std::cout
#include <string>     // std::string, std::stoi
using namespace cv;
using namespace std;


Mat sobel(Mat gray);
Mat canny(Mat src);






#define DEBUG_MODE_OFF (0)
#define DEBUG_MODE_ON (1)
#define BACKGROND_COLOR_CHANNELS (3)
#define BACKGROND_COLOR_RED (0)
#define BACKGROND_COLOR_GREEN (1)
#define BACKGROND_COLOR_BLUE (2)

#define RESULT_FILE_WORDS (200)
//#define NumOfcluster (4) /* number of cluster */
#define DEBUG_MODE (0)/*If u input 0,this program runs as runnning mode. 1 is debug mode*/

using namespace std;

/*function DebugClusterNumber*/
int DebugClusterNumber(int Number){
	
		return Number;
	
}



int main()
{
	/*
	int NumOfcluster;

	cout << ("\nEnter a num of clusters :\n");
	cin >> NumOfcluster;

	int i, j, size;
	IplImage *src_img = 0, *dst_img = 0;
	CvMat tmp_header;
	CvMat *clusters, *points, *tmp;
	CvMat *count = cvCreateMat(NumOfcluster, 1, CV_32SC1);
	CvMat *centers = cvCreateMat(NumOfcluster, 3, CV_32FC1);
	//set background color
	int background_color[BACKGROND_COLOR_CHANNELS] = { 255, 135, 60 };

	//set place u save image file.
	char file_name[] = "";
	char file_extension[] = ".bmp";

	// (1)load a specified file as a 3-channel color image


	char *imagename = "Photos\\wpp.jpg";




	printf("\ngot path: %s", imagename);

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
	cvKMeans2(points, NumOfcluster, clusters,
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
		cvNamedWindow("src", CV_GUI_NORMAL);
		cvShowImage("src", src_img);
		cvNamedWindow("low-color", CV_GUI_NORMAL);
		cvShowImage("low-color", dst_img);
		cvWaitKey(0);
		cvDestroyWindow("src");
		cvDestroyWindow("low-color");
	}
	//If u selected DEBUG_MODE_ON, cluster which u selected outputs.
	//cluster which u didn't select is painted black color.
	else if (DEBUG_MODE == DEBUG_MODE_ON){
		for (j = 0; j<NumOfcluster; j++)
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
//	cvReleaseImage(&src_img);
//	cvReleaseImage(&dst_img);
	//cvReleaseMat(&clusters);
	//cvReleaseMat(&points);
	//cvReleaseMat(&count);
	*/
	// https://github.com/LowWeiLin/OpenCV_ImageBackgroundRemoval/tree/master/OpenCV_ImageBackgroundRemoval
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// KMEANS ENDS AND BACKGROUND BLACKNING STARTS//////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//Load source image	
	//Mat src = imread("human-t-pose.jpg");
	//Mat src = imread("person1.jpg");
	Mat src = imread("Photos\\wpp1.jpg");

	//Mat src = dst_img; //////////////////////// TAKE THE OUTPUT OF THE KMEANS AND PUT INSIDE BACKGROUND REMOVAL

	//windows
	
	namedWindow("src", WINDOW_NORMAL);
	namedWindow("1. \"Remove Shadows\"", WINDOW_NORMAL);
	namedWindow("2. Grayscale", WINDOW_NORMAL);
	namedWindow("3. Edge Detector", WINDOW_NORMAL);
	namedWindow("4. Dilate", WINDOW_NORMAL);
	namedWindow("5. Floodfill", WINDOW_NORMAL);
	namedWindow("6. Erode", WINDOW_NORMAL);
	namedWindow("7. Largest Contour", WINDOW_NORMAL);
	namedWindow("8. Masked Source", WINDOW_NORMAL);
	namedWindow("src boxed", WINDOW_NORMAL);
	

	//0. Source Image
	imshow("src", src);

	//1. Remove Shadows
	//Convert to HSV
	Mat hsvImg;
	cvtColor(src, hsvImg, CV_BGR2HSV);
	Mat channel[3];
	split(hsvImg, channel);
	channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
	//Merge channels
	merge(channel, 3, hsvImg);
	Mat rgbImg;
	cvtColor(hsvImg, rgbImg, CV_HSV2BGR);
	imshow("1. \"Remove Shadows\"", rgbImg);

	//2. Convert to gray and normalize
	Mat gray(rgbImg.rows, src.cols, CV_8UC1);
	cvtColor(rgbImg, gray, CV_BGR2GRAY);
	normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("2. Grayscale", gray);

	//3. Edge detector
	GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat edges;
	bool useCanny = false;
	if (useCanny){
		edges = canny(gray);
	}
	else {
		//Use Sobel filter and thresholding.			
		edges = sobel(gray);
		//Automatic thresholding
		threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
		//Manual thresholding
		//threshold(edges, edges, 25, 255, cv::THRESH_BINARY);
	}

	imshow("3. Edge Detector", edges);

	//4. Dilate
	Mat dilateGrad = edges;
	int dilateType = MORPH_ELLIPSE;
	int dilateSize = 3;
	Mat elementDilate = getStructuringElement(dilateType,
		Size(2 * dilateSize + 1, 2 * dilateSize + 1),
		Point(dilateSize, dilateSize));
	dilate(edges, dilateGrad, elementDilate);
	imshow("4. Dilate", dilateGrad);

	//5. Floodfill
	Mat floodFilled = cv::Mat::zeros(dilateGrad.rows + 2, dilateGrad.cols + 2, CV_8U);
	floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
	floodFilled = cv::Scalar::all(255) - floodFilled;
	Mat temp;
	floodFilled(Rect(1, 1, dilateGrad.cols - 2, dilateGrad.rows - 2)).copyTo(temp);
	floodFilled = temp;
	imshow("5. Floodfill", floodFilled);

	//6. Erode
	int erosionType = MORPH_ELLIPSE;
	int erosionSize = 4;
	Mat erosionElement = getStructuringElement(erosionType,
		Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		Point(erosionSize, erosionSize));
	erode(floodFilled, floodFilled, erosionElement);
	imshow("6. Erode", floodFilled);

	//7. Find largest contour
	int largestArea = 0;
	int largestContourIndex = 0;
	Rect boundingRectangle;
	Mat largestContour(src.rows, src.cols, CV_8UC1, Scalar::all(0));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i<contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largestArea)
		{
			largestArea = a;
			largestContourIndex = i;
			boundingRectangle = boundingRect(contours[i]);
		}
	}

	Scalar color(255, 255, 255);
	drawContours(largestContour, contours, largestContourIndex, color, CV_FILLED, 8, hierarchy); //Draw the largest contour using previously stored index.
	rectangle(src, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);
	imshow("7. Largest Contour", largestContour);

	//8. Mask original image
	Mat maskedSrc;
	src.copyTo(maskedSrc, largestContour);
	imshow("8. Masked Source", maskedSrc);

	//Source with largest contour boxed
	imshow("src boxed", src);


	waitKey(0);
}

Mat sobel(Mat gray){
	Mat edges;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat edges_x, edges_y;
	Mat abs_edges_x, abs_edges_y;
	Sobel(gray, edges_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(edges_x, abs_edges_x);
	Sobel(gray, edges_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(edges_y, abs_edges_y);
	addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, edges);

	return edges;
}

Mat canny(Mat src)
{
	Mat detected_edges;

	int edgeThresh = 1;
	int lowThreshold = 250;
	int highThreshold = 750;
	int kernel_size = 5;
	Canny(src, detected_edges, lowThreshold, highThreshold, kernel_size);

	return detected_edges;
}