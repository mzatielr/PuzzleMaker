#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <direct.h>

#include <iostream>
#include <thr/xthrcommon.h>
#include <ctime>
/* time_t, struct tm, difftime, time, mktime */

using namespace cv;
using namespace std;

// Edge Detection area
/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
bool isUIMode = false;

const double alpha = 1.0 / 255.0; // Don't change const value
const int K = 7;

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


void RemoveClusters(Mat& src)
{
	Rect* rect = new Rect();
	double alpha = 1.0 / 255.0;
	Mat mask = Mat::zeros(src.rows + 2, src.cols + 2, CV_8UC1);

	// East
	for (int i = 0; i < src.rows; ++i)
	{
		floodFill(src, mask, Point(0, i), 0, rect, cvScalarAll(0), cvScalarAll(0), 4 | (255 << 8));
	}

	// West
	for (int i = 0; i < src.rows; ++i)
	{
		floodFill(src, mask, Point(src.cols - 1, i), 0, rect, cvScalarAll(0), cvScalarAll(0), 4 | (255 << 8));
	}

	// North
	for (int i = 0; i < src.cols; ++i)
	{
		floodFill(src, mask, Point(i, 0), 0, rect, cvScalarAll(0), cvScalarAll(0), 4 | (255 << 8));
	}

	// South
	for (int i = 0; i < src.cols; ++i)
	{
		floodFill(src, mask, Point(i, src.rows - 1), 0, rect, cvScalarAll(0), cvScalarAll(0), 4 | (255 << 8));
	}
	imwrite(path1+"Mask.bmp", mask);

	ShowImage(src, "afterClusterRemoval");
	ShowImage(mask, "Mask");

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
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

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


//if we cant take the serounding of the pixle we will return 0, if we can will return 1
int GenerateKernel(const Mat& src, int i, int j, Mat& kernel)
{


	const int rectEdge = GetRectEdge();

	int shift = (rectEdge - 1) / 2;

	if (i - shift < 0)
	{
	//	cout << i - shift << "< 0 \n";
		return 0;
	}
	if (j - shift < 0)
	{
	//	cout << j - shift << "< 0 \n";
		return 0;
	}
	if (i + shift >= src.rows)
	{
	//	cout << i + shift << ">=" << src.rows<<"\n";
		return 0;
	}
	if (j + shift >= src.cols)
	{
	//	cout << j + shift << ">=" << src.cols<<"\n";

		return 0;
	}

	int cnt = 0;

	int a1 =  i - shift;
	int a2 =  i + shift;
	int b1 =  j - shift;
	int b2 =  j + shift;

	/*
	int a1 = max(0, i - shift);
	int a2 = min(src.rows, a1 + rectEdge);
	
	if (a1 + rectEdge >= src.rows)
	{
		
		int t = (src.rows - (a1 + rectEdge));
		a1 -= t;
	}

	int b1 = max(0, j - shift);
	int b2 = min(src.cols, b1 + rectEdge);
	
	if (b1 + rectEdge >= src.cols)
	{
		cout << a1 << ">=" << src.rows << '\n';

		return 0;
		//int t = (src.cols - (b1 + rectEdge));
		//b1 -= t;
	}
	//cout << "a2: " << a2 << " a1: " << a1 << " b2: " << b2 << " b1: " << b1 << endl;

	*/

 	//CV_Assert(a2 - a1 == rectEdge && b2 - b1 == rectEdge);
	for (int row = a1; row < a2; ++row)
	{
		for (int col = b1; col < b2; ++col)
		{
			kernel.at<float>(cnt, 0) = static_cast<float>(src.at<Vec3f>(row, col).val[0]);
			kernel.at<float>(cnt, 1) = static_cast<float>(src.at<Vec3f>(row, col).val[1]);
			kernel.at<float>(cnt, 2) = static_cast<float>(src.at<Vec3f>(row, col).val[2]);
			cnt++;
		}
	}
	return 1;
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
	cout << endl;
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
	return a * b;
}

void FillDataInPcaInput(Mat& targetMat, int global_cnt, const Mat& kernel)
{
	int colCnt = 0;
	// Todo: Implementation can be improved using mat::reshape
	for (int i = 0; i < kernel.rows; ++i)
	{
		for (int j = 0; j < kernel.cols; ++j)
		{
			targetMat.at<float>(global_cnt, colCnt) = kernel.at<float>(i, j);
			colCnt++;
		}
	}
}

// Input: 
//		rectEdge - rect size for pca evaluation 
//		srcConverterd - Type: CV_32FC3, src image to calculate pca new image
// Output:
//		imageAfterPca - Type: CV_32FC3, initialized to zeros Mat with equal size as src image
void ExecutePCA(const int rectEdge, const Mat& srcConverterd, Mat& imageAfterPca)
{
	double pcaCoreTimeCounter = 0;
	time_t temp1, temp2;
	time_t time1, time2;
	PCA pca_analysis;
	int numOfDiscardedPixels = 0;
	time(&time1);
	Mat pcaInput = Mat::zeros(srcConverterd.rows * srcConverterd.cols, rectEdge * rectEdge * 3, CV_32FC1);
	int kernelGenResult;
	auto emptyArray = noArray();
	int globalCnt = 0;
	for (int i = 0; i < srcConverterd.rows; ++i)
	{
		for (int j = 0; j < srcConverterd.cols; ++j)
		{
			//int kernelSize = GetKernelSize(rectEdge, i, j, srcConverterd.rows, srcConverterd.cols);
			Mat kernel = Mat::zeros(pcaInput.cols, 3, CV_32FC1);
			kernelGenResult = GenerateKernel(srcConverterd, i, j, kernel);
			if (kernelGenResult == 1)
			{
				FillDataInPcaInput(pcaInput, globalCnt, kernel);
				globalCnt++; // IS IT THE PRObLeM ?
			}
			else
			{ 
				numOfDiscardedPixels++;
		//		cout << i << " "<< j<<'\n'; 
			
			}
			//PrintMatC1(kernel);
		}
	}
	cout <<"discarded "<< numOfDiscardedPixels << " out of " << srcConverterd.cols * srcConverterd.rows<<"\n";
	time(&time2);

	//pca
	//Perform PCA analysis
	time(&temp1);
	pca_analysis(pcaInput, emptyArray, CV_PCA_DATA_AS_ROW, 3); // reduction to 3 dimensions
	time(&temp2);
	pcaCoreTimeCounter += difftime(temp2, temp1);
	//Mat meanPoint = pca_analysis.mean;
	Mat projection_result;

	pca_analysis.project(pcaInput, projection_result);
	CV_Assert(3 == projection_result.cols && projection_result.rows == pcaInput.rows);

	//ATTENTION construction should start and end on shifted pixels !!
	/*
	for (int i = 2; i < projection_result.rows; ++i)
	{
		int reducedCols = imageAfterPca.cols - 5;
		imageAfterPca.at<Vec3f>(i / imageAfterPca.cols, i % reducedCols).val[0] = projection_result.at<float>(i, 0);
		imageAfterPca.at<Vec3f>(i / imageAfterPca.cols, i % reducedCols).val[1] = projection_result.at<float>(i, 1);
		imageAfterPca.at<Vec3f>(i / imageAfterPca.cols, i % reducedCols).val[2] = projection_result.at<float>(i, 2);
	}
	*/
	int shift = rectEdge / 2; 
	int count = 0;
	for (int i = 0; i < srcConverterd.rows - shift * 2; i++)
	{

		for (int j = 0; j < srcConverterd.cols - shift*2; j++)
		{
			imageAfterPca.at<Vec3f>(i, j).val[0] = projection_result.at<float>(count, 0);
			imageAfterPca.at<Vec3f>(i, j).val[1] = projection_result.at<float>(count, 1);
			imageAfterPca.at<Vec3f>(i, j).val[2] = projection_result.at<float>(count, 2);
			count++;
		}

	}

	std::cout << "All iteration took " << difftime(time2, time1) << endl;
	cout << "PCA analysis took " << pcaCoreTimeCounter << endl;

	imageAfterPca.convertTo(imageAfterPca, CV_8UC3, 255.0);
	cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

//	cout << pcaInput;
	cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

	ShowImage(imageAfterPca, "8UImageAfterPCA");
	imwrite(path1+"ImageAfterPCA.bmp", imageAfterPca);
	imageAfterPca.convertTo(imageAfterPca, CV_32FC3, alpha);
}

// Input: 
//		imageAfterPca- Type: CV_32FC3, src image to calculate kmeans input mat
// Output:
//		kmeansMat - KMeans imput mat. Initialized to:  Mat::zeros(imageAfterPca.rows*imageAfterPca.cols, 1, CV_32FC3);
void GenerateKmeansInputMat(const Mat& imageAfterPca, Mat& kmeansMat)
{
	int counter = 0;
	for (int i = 0; i < imageAfterPca.rows; i++)
	{
		for (int j = 0; j < imageAfterPca.cols; ++j)
		{
			kmeansMat.at<Vec3f>(counter, 0)[0] = imageAfterPca.at<Vec3f>(i, j)[0];
			kmeansMat.at<Vec3f>(counter, 0)[1] = imageAfterPca.at<Vec3f>(i, j)[1];
			kmeansMat.at<Vec3f>(counter, 0)[2] = imageAfterPca.at<Vec3f>(i, j)[2];
			counter++;
		}
	}
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


	src1.convertTo(src1, CV_32FC3, alpha);
	//PrintMatDouble(srcConverterd);

	//srcConverterd.convertTo(srcConverterd, CV_8UC3, 255.0);
	//PrintMatShort(srcConverterd);

	//imageAfterPCA will be smaller then the original one because image edges were trimmed !!
	Mat imageAfterPca = Mat::zeros(src1.rows - (rectEdge -1) , src1.cols - (rectEdge-1 ), CV_32FC3);
	cout << "Starting pca" << endl;
	ExecutePCA(rectEdge, src1, imageAfterPca);
	cout << "End of PCA stage" << endl;
	//ShowImage(imageAfterPca, "DoubleImageAfterPCA");
	//imageAfterPca.convertTo(imageAfterPca, CV_8UC3, 255.0);

	Mat kmeansMat = Mat::zeros(imageAfterPca.rows * imageAfterPca.cols, 1, CV_32FC3);

	GenerateKmeansInputMat(imageAfterPca, kmeansMat);

	cout << "Executing kmeans" << endl;
	Mat bestLabels, centers;

	kmeans(kmeansMat, K, bestLabels,
	       TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 10, 1.0), 1, KMEANS_PP_CENTERS, centers);

	cout << "Finish to execute kmeans" << endl;

	int colors[K];
	for (int i = 0; i < K; i++)
	{
		colors[i] = 255 / (i + 1);
	}

	Mat clustered = Mat(imageAfterPca.rows, imageAfterPca.cols, CV_32F);
	for (int i = 0; i < imageAfterPca.cols * imageAfterPca.rows; i++)
	{
		clustered.at<float>(i / imageAfterPca.cols, i % imageAfterPca.cols) = static_cast<float>(colors[bestLabels.at<int>(0, i)]);
	}

	clustered.convertTo(clustered, CV_8U);
	ShowImage(clustered, "clustered");


	imwrite(path1+"ClusterdImage.bmp", clustered);

	RemoveClusters(clustered);

	imwrite(path1+"AfterClusterRemoval.bmp", clustered);
}

void KMeans(string path)
{

	size_t position = path.find(".");


	path1 = path.substr(0,position);

	cout << "Results saved at: " << path1 <<'\n';


	Mat src1 = imread(path);


	_mkdir((path1).c_str());

	path1 = path1 + "\\";

	imwrite(path1 + "orig.bmp", src1);

	if (src1.empty())
	{
		cout << "ATTENTION: no image loaded! Stopping Execution\n";
		return;
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
	for (int i = 1; i < argc; ++i)
	{
		string currentImage = argv[i];
		cout << "Handling follwing image " << currentImage << endl;
 		KMeans(currentImage);
	}
	
	//KMeans("C:\\oldDesktop\\סדנה\\KMeansConsoleApplication\\KMeansConsoleApplication\\nonwhite.bmp");
	waitKey();
	destroyAllWindows();

	return 0;
}
