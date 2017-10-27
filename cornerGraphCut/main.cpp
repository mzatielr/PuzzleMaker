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


vector<Point> cornerHarris_demo( Mat& src)
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.045;
	Mat src_gray;
	char* corners_window = "Corners detected";
	int thresh = 500;
	
	// Convert src image to gray color
	cvtColor(src, src_gray, CV_BGR2GRAY);
	imwrite("gray.bmp", src_gray);


	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	imwrite("dst_norm_scaled.bmp", dst_norm_scaled);

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
	void showImage() const;
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
void GCApplication::showImage() const
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
		
	imshow(*winName, res);
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

	int edgeSize = 65;

	Point brightest = getBrightPoint(*image);

	rect = Rect(Point(MidCornerGlobalPoint.x - edgeSize, MidCornerGlobalPoint.y - edgeSize), Point(MidCornerGlobalPoint.x + edgeSize, MidCornerGlobalPoint.y + edgeSize));

	
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
int main(int argc, char** argv)
{
	ExecutePreImageProcessing("orig5.jpg");
	
	//_CrtDbgBreak();
	help();

	string filename = "pca.jpg";
	if (filename.empty())
	{
		cout << "\nDurn, empty filename" << endl;
		return 1;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}
	const string winName = "image";
	namedWindow(winName, WINDOW_AUTOSIZE);

	MidCornerGlobalPoint = midPointFromCollection(cornerHarris_demo(image));

	setMouseCallback(winName, on_mouse, 0);
	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	for (;;)
	{
		char c = static_cast<char>(waitKey(0));
		switch (c)
		{
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':
			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}
exit_main:
	destroyWindow(winName);
	return 0;
}