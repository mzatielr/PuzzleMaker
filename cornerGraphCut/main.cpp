#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <math.h>
#include "pca.h"

using namespace std;
using namespace cv;
static void help()
{
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set GC_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
}
const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;
Point MidCornerGlobalPoint;
string relativeImageFolderPath;

void ImageWrite(const string& str, const Mat& mat)
{
	imwrite(relativeImageFolderPath + str, mat);
}

void ShowImageUI(const Mat& src, const string& name)
{
	namedWindow(name, WINDOW_NORMAL);
	imshow(name, src);
}

int goodFeaturesToTrackMethod(const Mat& src )
{
	std::vector< cv::Point2f > corners;

	// maxCorners – The maximum number of corners to return. If there are more corners
	// than that will be found, the strongest of them will be returned
	int maxCorners = 10;

	// qualityLevel – Characterizes the minimal accepted quality of image corners;
	// the value of the parameter is multiplied by the by the best corner quality
	// measure (which is the min eigenvalue, see cornerMinEigenVal() ,
	// or the Harris function response, see cornerHarris() ).
	// The corners, which quality measure is less than the product, will be rejected.
	// For example, if the best corner has the quality measure = 1500,
	// and the qualityLevel=0.01 , then all the corners which quality measure is
	// less than 15 will be rejected.
	double qualityLevel = 0.01;

	// minDistance – The minimum possible Euclidean distance between the returned corners
	double minDistance = 20;

	// mask – The optional region of interest. If the image is not empty (then it
	// needs to have the type CV_8UC1 and the same size as image ), it will specify
	// the region in which the corners are detected
	cv::Mat mask;

	// blockSize – Size of the averaging block for computing derivative covariation
	// matrix over each pixel neighborhood, see cornerEigenValsAndVecs()
	int blockSize = 3;

	// useHarrisDetector – Indicates, whether to use operator or cornerMinEigenVal()
	bool useHarrisDetector = false;

	// k – Free parameter of Harris detector
	double k = 0.04;

	cv::goodFeaturesToTrack(src, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

	for (size_t i = 0; i < corners.size(); i++)
	{
		cv::circle(src, corners[i], 2, cv::Scalar(255.), -1);
	}

	string windowName = "GoodFeatureDetector";
	cv::namedWindow(windowName, CV_WINDOW_NORMAL);
	cv::imshow(windowName, src);

	return EXIT_SUCCESS;
}


vector<Point> cornerHarris_demo( const Mat& src)
{

	Mat dst_norm, dst_norm_scaled;
	Mat dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	Mat src_gray;
	char* corners_window = "Corners detected";
	int thresh = 500;
	
	// Convert src image to gray color
	cvtColor(src, src_gray, CV_BGR2GRAY);
	imwrite(relativeImageFolderPath + "gray.bmp", src_gray);

	goodFeaturesToTrackMethod(src_gray.clone());

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	
	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	ShowImageUI(dst_norm_scaled, "dst_norm");
	ImageWrite("dst_norm.jpg", dst_norm_scaled);

	std::vector<Point> pointCollection;
	while (pointCollection.size() < 20)
	{
		/// Drawing a circle around corners
		for (int j = 0; j < dst_norm.rows; j++)
		{
			for (int i = 0; i < dst_norm.cols; i++)
			{
				if ((int)dst_norm.at<float>(j, i) > thresh)
				{
					pointCollection.push_back(Point(i, j));
					circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
				}
			}
		}

		if (!pointCollection.empty())
			thresh--;
		else
			thresh -= 10;

		cout << "found " << pointCollection.size() << " corrners\n";
	}
	if (pointCollection.size() < 2)
	{
		cout << "Cant find 2 corners";
	}

	/// Showing the result
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
	imwrite(relativeImageFolderPath + "corner.jpg", dst_norm_scaled);
	return pointCollection;
}

int euclidDistance(Point a, Point b)
{
	return  sqrt(pow( (a.x-b.x), 2) + pow( (a.y-b.y), 2));

}
Point midPoint(Point a, Point b)
{
	return Point(  floor((a.x + b.x) / 2), floor((a.y + b.y) / 2));
}
Point midPointFromCollection(vector<Point> collection)
{
	int m;
	int maxDist = 0;
	Point midPointResult;
	for (int i = 0; i < collection.size(); i++)
	{
		for (int j = 0; j < collection.size(); j++)
		{
			m = euclidDistance(collection[i], collection[j]);
			if (m>maxDist)
			{
				maxDist = m;
				midPointResult = midPoint(collection[i], collection[j]);
			}
		}


	}
	return midPointResult;
}

static Point getBrightPoint(const Mat& img)
{
	// RGB values of the color you are looking for
	int r = 255;
	int g = 0;
	int b = 255;

	// Load image
	// Split image into channels
	cv::Mat channels[3];
	cv::split(img, channels);

	// Find absolute differences for each channel
	cv::Mat diff_r;
	cv::absdiff(channels[2], r, diff_r);
	cv::Mat diff_g;
	cv::absdiff(channels[1], g, diff_g);
	cv::Mat diff_b;
	cv::absdiff(channels[0], b, diff_b);

	// Calculate L1 distance
	cv::Mat dist = diff_r + diff_g + diff_b;

	// Find the location of pixel with minimum color distance
	cv::Point minLoc;
	cv::minMaxLoc(dist, 0, 0, &minLoc);

	// Get the color of a pixel at minLoc
	return minLoc;

	
}
static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;
}
class GCApplication
{
public:
	enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;
	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage(string folderToSaveImage = "", bool isSave = false) const;
	void showImageUI() const;
	void simulateSelectRect(int rectType = 0);
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);
	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;
	uchar rectState, lblsState, prLblsState;
	bool isInitialized;
	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
};
void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();
	isInitialized = false;
	rectState = NOT_SET;
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}
void GCApplication::showImage(string folderToSaveImage, bool isSave) const
{
	if (image->empty() || winName->empty())
		return;
	Mat res;
	Mat binMask;
	if (!isInitialized)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		image->copyTo(res, binMask);
	}
	vector<Point>::const_iterator it;
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
		circle(res, *it, radius, RED, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);
	if (rectState == IN_PROCESS || rectState == SET)
	{
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
		rectangle(res, MidCornerGlobalPoint, MidCornerGlobalPoint, GREEN, 2);

	}
	if (isSave) imwrite(folderToSaveImage + "res.jpg", res);
	imshow(*winName, res);
}

void GCApplication::showImageUI() const
{
	ShowImageUI(*image, "Image");
}

void GCApplication::simulateSelectRect(int rectType)
{
	int rows = image->rows;
	int cols = image->cols;

	int x, y, width, hight;
	
	switch (rectType)
	{
	case 1: //Right Half
		x = cols/2, y = 2, width = cols - 10, hight = rows - 10;
		break;
	case 2: //Right Half
		x = 2, y = 2, width = cols/2, hight = rows - 10;
		break;
	default:
		x = 2, y = 2, width = cols - 10, hight = rows - 10;
		break;
	}


	rect = Rect(x, y, width, hight);
	rectState = SET;
	setRectInMask();
	CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
	showImage();
}

void GCApplication::setRectInMask()
{
	CV_Assert(!mask.empty());
	mask.setTo(GC_BGD);
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}
void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr)
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	else
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD;
		fvalue = GC_PR_FGD;
	}
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);
	}
}
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check
	switch (event)
	{
	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) && rectState == SET)
			lblsState = IN_PROCESS;
	}
	break;
	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET)
			prLblsState = IN_PROCESS;
	}
	break;
	case EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			rectState = SET;
			setRectInMask();
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			lblsState = SET;
			showImage();
		}
		break;
	case EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			prLblsState = SET;
			showImage();
		}
		break;
	case EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		else if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

int GCApplication::nextIter()
{

	int edgeSize = 300;

	Point brightest = getBrightPoint(*image);
	int a = max(0, MidCornerGlobalPoint.x - edgeSize);
	int b = max(0, MidCornerGlobalPoint.y - edgeSize);

	int c = min(MidCornerGlobalPoint.x + edgeSize, image->cols - 1);
	int d = min(MidCornerGlobalPoint.y + edgeSize, image->rows - 1);
	//rect = Rect(Point(a, b), Point(c, d));

	cout << rect << endl;
	
	if (isInitialized)
		grabCut(*image, mask, rect, bgdModel, fgdModel, 1);
	else
	{
		if (rectState != SET)
			return iterCount;
		if (lblsState == SET || prLblsState == SET)
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		else
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();
	return iterCount;
}
GCApplication gcapp;
static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}
static void DestroyWindows()
{
	destroyAllWindows();
}

static void NextIterationAction()
{
	int iterCount = gcapp.getIterCount();
	cout << "<" << iterCount << "... ";
	int newIterCount = gcapp.nextIter();
	gcapp.showImage(relativeImageFolderPath, true);
}

void IterationRunner(int num_of_grub_cut_itration)
{
	for (int i = 0; i < num_of_grub_cut_itration; ++i)
	{
		NextIterationAction();
	}
}

int ImageHandler(const string& cs)
{
	bool isUsePCA = false;
	bool isSimulate = true;

	if (isUsePCA)
		ExecutePreImageProcessing(cs);

	//_CrtDbgBreak();
	help();

	int pos = cs.find(".");
	relativeImageFolderPath = cs.substr(0, pos) + "\\";
	string filename;
	if (isUsePCA) 
		filename = relativeImageFolderPath + "pca.jpg";
	else
		filename = relativeImageFolderPath + cs;

	if (filename.empty())
	{
		cout << "\nDurn, empty filename" << endl;
		return 1;
	}

	Mat image = imread(cs);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}
	const string winName = "image";
	namedWindow(winName, WINDOW_AUTOSIZE);

	MidCornerGlobalPoint = midPointFromCollection(cornerHarris_demo(image.clone()));

	setMouseCallback(winName, on_mouse, nullptr);
	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	
	int numOfGrubCutItration = 20;

	if (isSimulate)
	{
		on_mouse(EVENT_LBUTTONDOWN, 2, 2, 1, nullptr);
		on_mouse(EVENT_LBUTTONUP, image.cols - 10, image.rows - 10, 0, nullptr);

		IterationRunner(numOfGrubCutItration);
		goto exit_main;
	}

	for (;;)
	{
		char c = static_cast<char>(waitKey(0));
		switch (c)
		{
		case 'n':
			IterationRunner(1);
			break;
		default:
			goto exit_main;
		}
	}
exit_main:
	DestroyWindows();
	return 0;
}



int main(int argc, char** argv)
{
	for (int i = 1; i < argc; ++i)
	{
		string imagePath =  argv[i];
		cout << "Starting handle following image: " << imagePath << endl;
		ImageHandler(imagePath);
	}
	return 0;
}
