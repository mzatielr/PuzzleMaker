
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <thr/xthrcommon.h>

using namespace cv;
using namespace std;

// Edge Detection area
/// Global variables
int neutralColor ;
Mat src1; //will allways contain the original image ONLY
Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
bool isUIMode = true;

string path1;
string path2;
string path3;

void ShowImage(const Mat& src, const string& name)
{
	if (isUIMode)
	{
		namedWindow(name, WINDOW_NORMAL);
		imshow(name, src);
	}
}
//function to blacken the background from the ORIGINAL image
//todo: not working !! 
void blackenOrigImage(Mat& blackenClustered)
{
	setbuf(stdout, NULL);
	double alpha = 1.0 / 255.0;
	Mat origBlacken = src1.clone();
	origBlacken.convertTo(origBlacken, CV_32FC3, alpha);

	for (int i = 0; i < blackenClustered.rows; ++i)
	{
		for (int j = 0; j < blackenClustered.cols; ++j)
		{
			//cout << i << " " << j << "\n";
			if (blackenClustered.at<uchar>(Point(i,j )) == neutralColor)// access violation thrown at this line, why ?
			{
				origBlacken.at<Vec3b>(1, 1).val[0] =0 ;
				//cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
				origBlacken.at<Vec3d>(i, j).val[1] = 0;
				origBlacken.at<Vec3d>(i, j).val[2] = 0;
			}
		}

	}

	ShowImage(origBlacken, "finalOutPut\n");
}

void RemoveClusters(Mat& src)
{
	cout << "Starting cluster removal...\n";
	Rect* rect= new Rect();
	double alpha = 1.0 / 255.0;
	Mat mask = Mat::zeros(src.rows + 2, src.cols + 2, CV_8UC1);


	neutralColor = 1;
	// East
	for (int i = 0; i < src.rows; ++i)
	{
		floodFill(src, mask, Point(0, i), neutralColor, rect, cvScalarAll(0), cvScalarAll(0), 8 | (255 << 8));
	}

	// West
	for (int i = 0; i < src.rows; ++i)
	{
		floodFill(src, mask, Point(src.cols - 1, i), neutralColor, rect, cvScalarAll(0), cvScalarAll(0), 8 | (255 << 8));
	}

	// North
	for (int i = 0; i < src.cols; ++i)
	{
		floodFill(src, mask, Point(i, 0), neutralColor, rect, cvScalarAll(0), cvScalarAll(0), 8 | (255 << 8));
	}

	// South
	for (int i = 0; i < src.cols; ++i)
	{
		floodFill(src, mask, Point(i, src.rows - 1), neutralColor, rect, cvScalarAll(0), cvScalarAll(0), 8 | (255 << 8));
	}
	cout << "Finished cluster removal\n";

	imwrite(path3, mask);
	ShowImage(src, "afterClusterRemoval");
	ShowImage(mask, "Mask");

	blackenOrigImage(src);



	return;
}

/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	ShowImage(dst, window_name);
}


/** @function main */
void EdgeDetector(string path)
{
	src = imread(path);


	cout << "Starting endge detetion process" << endl;
	if (!src.data)
	{
		cout << "Error in parsind data from src image" << endl;
		return;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, WINDOW_NORMAL);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	/// Show the image
	CannyThreshold(0, 0);
}


int GetRectEdge()
{
	return 7;
}

void GenerateKernel(const Mat& src, int i, int j, Mat& kernel)
{
	const int rectEdge = GetRectEdge();

	int shift = (rectEdge - 1) / 2;
	int cnt = 0;
	for (int row = max(0, i - shift); row <= min(src.rows - 1, i + shift); ++row)
	{
		for (int col = max(0, j - shift); col <= min(src.cols - 1, j + shift); ++col)
		{
			kernel.at<double>(cnt, 0) = static_cast<double>(src.at<Vec3d>(row, col).val[0]);
			kernel.at<double>(cnt, 1) = src.at<Vec3d>(row, col).val[1];
			kernel.at<double>(cnt, 2) = src.at<Vec3d>(row, col).val[2];
			cnt++;
		}
	}
}

void PrintMatShort(const Mat& mat)
{
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			cout << "(" << static_cast<short>(mat.at<Vec3b>(i, j).val[0]) << ", " << (short)mat.at<Vec3b>(i, j)[1] << ", " << (short)mat.at<Vec3b>(i, j)[2] << ") ";
		}
		cout << endl;
	}
	cout << endl;
}

void PrintMatDouble(const Mat& mat)
{
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			cout << "(" << static_cast<double>(mat.at<Vec3d>(i, j).val[0]) << ", " << (double)mat.at<Vec3d>(i, j)[1] << ", " << (double)mat.at<Vec3d>(i, j)[2] << ") ";
		}
		cout << endl;
	}
	cout  << endl;
}

void PrintMatC1(const Mat& mat)
{
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			cout << "(" << static_cast<double>(mat.at<double>(i, j)) << ") ";
		}
		cout << endl;
	}
	cout << endl;
}

// i - current row
// j - current col
int GetKernelSize(const int rect_edge, int i, int j, int rows, int cols)
{
	int shift = (rect_edge - 1) / 2;
	int a = min(rows - 1, i + shift) - max(0, i - shift) + 1;
	int b = min(cols - 1, j + shift) - max(0, j - shift) + 1;
	return a*b;
}

void GenerateNeightMatrix(Mat& src1)
{
	const int rectEdge = GetRectEdge();

	cout << "Starting pre stage" << endl;
	ShowImage(src1, "original");

	//GaussianBlur(src1, src1, Size(5, 5), 0, 0);
	//blur(src, src, Size(3,3));

	//ShowImage(src1, "blurred");
	//PrintMatShort(src1);
	Mat origBlaken = src1.clone();

	Mat srcConverterd = src1.clone();
	double alpha = 1.0 / 255.0;
	cout << alpha << endl;
	srcConverterd.convertTo(srcConverterd, CV_64FC3, alpha);
	//PrintMatDouble(srcConverterd);

	//srcConverterd.convertTo(srcConverterd, CV_8UC3, 255.0);
	//PrintMatShort(srcConverterd);

	Mat p = Mat::zeros(src1.cols*src1.rows,  3, CV_64FC1);
	Mat imageAfterPca = Mat::zeros(src1.rows, src1.cols, CV_64FC3);

	for (int i = 0; i < srcConverterd.rows; ++i)
	{
		for (int j = 0; j < srcConverterd.cols; ++j)
		{
			int kernelSize = GetKernelSize(rectEdge, i, j, srcConverterd.rows, srcConverterd.cols);
			Mat kernel = Mat::zeros(kernelSize, 3, CV_64FC1);
			GenerateKernel(srcConverterd, i, j, kernel);

			//PrintMatC1(kernel);
			
			//pca
			//Perform PCA analysis
			PCA pca_analysis(kernel, noArray(), CV_PCA_DATA_AS_ROW);
			Mat meanPoint = pca_analysis.mean;
			//cout << "Mean: rowsXcols " << meanPoint.rows << "X" << meanPoint.cols << endl;
			//cout << meanPoint << endl;

			imageAfterPca.at<Vec3d>(i, j).val[0] = meanPoint.at<double>(0, 0);
			imageAfterPca.at<Vec3d>(i, j).val[1] = meanPoint.at<double>(0, 1);
			imageAfterPca.at<Vec3d>(i, j).val[2] = meanPoint.at<double>(0, 2);

			// same mean in p
		}
	}

	//ShowImage(imageAfterPca, "DoubleImageAfterPCA");
	imageAfterPca.convertTo(imageAfterPca, CV_8UC3, 255.0);
	ShowImage(imageAfterPca, "8UImageAfterPCA");

	Mat kmeansInput = imageAfterPca.clone(); // with pca
	//Mat kmeansInput = src1.clone(); // without pca


	kmeansInput.convertTo(kmeansInput, CV_32FC3, alpha);

	Mat kmeansMat = Mat::zeros(kmeansInput.rows*kmeansInput.cols, 1, CV_32FC3);
	
	int counter = 0;
	for (int i = 0; i<kmeansInput.rows; i++)
	{
		for (int j = 0; j < kmeansInput.cols; ++j)
		{
			kmeansMat.at<Vec3f>(counter, 0)[0] = kmeansInput.at<Vec3f>(i, j)[0];
			kmeansMat.at<Vec3f>(counter, 0)[1] = kmeansInput.at<Vec3f>(i, j)[1];
			kmeansMat.at<Vec3f>(counter, 0)[2] = kmeansInput.at<Vec3f>(i, j)[2];
			counter++;
		}
	}

	cout << "Executing kmeans" << endl;
	Mat bestLabels, centers;

	int const K = 7;
	kmeans(kmeansMat, K, bestLabels,
		TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 10, 1.0), 1, KMEANS_PP_CENTERS, centers);

	cout << "Finish to execute kmeans" << endl;
	int colors[K];
	for (int i = 0; i<K; i++)
	{
		colors[i] = 255 / (i + 1);
	}

	Mat clustered = Mat(kmeansInput.rows, kmeansInput.cols, CV_32F);
	for (int i = 0; i<kmeansInput.cols*kmeansInput.rows; i++)
	{
		clustered.at<float>(i / kmeansInput.cols, i%kmeansInput.cols) = static_cast<float>(colors[bestLabels.at<int>(0, i)]);
	}

	clustered.convertTo(clustered, CV_8U);
	ShowImage(clustered, "clustered");

	RemoveClusters(clustered);

}

void KMeans(string path)
{
	src1 = imread(path);
	if (src1.empty())
	{
		cout << "ATTENTION: no image loaded!\n";
	}
	else
	{
		cout << "Loaded " << src1.rows << " X " << src1.cols << " image\n";
	}
	//RemoveClusters(src1);

	//imwrite("wspp1.bmp", src1);
	GenerateNeightMatrix(src1);
	//EdgeDetector(src);
	/// Wait until user exit program by pressing a key
}


int main(int argc, char** argv) 
{
	/*
	path1 = argv[2];
	path2 = argv[3];
	path3 = argv[4];
	KMeans(argv[1]);*/
	path1 = "a.jpg";
	path2 = "b.jpg";
	path3 = "c.jpg";
	KMeans("C:\\oldDesktop\\סדנה\\KMeansConsoleApplication\\KMeansConsoleApplication\\nonwhite.bmp");

	waitKey();
	destroyAllWindows();

    return 0;
}





