#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
const int Quad = 4;
static vector<Point2i> sQuadPoints;
static Point2i sCurrentPt;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if (sQuadPoints.size() == Quad)
			cout << "Cannot choose more point!" << std::endl;
		else {
			sQuadPoints.push_back(Point2i(x, y));
		}
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if (sQuadPoints.empty())
			cout << "Cannot remove more point!" << endl;
		else
			sQuadPoints.pop_back();
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		sCurrentPt = Point2i(x, y);
	}
}
void help()
{
	cout << "This binary is used to rectify image." << endl;
	cout << "Step 1. Move mouse and click left button to select four points for the target in order." << endl;
	cout << "Step 2. Click right button to cancel one point selection." << endl;
	cout << "Setp 3. After selecting four points, press ENTER to show the rectification result." << endl;
}
int main(int argc, char** argv)
{
	// Read image from file 
	if (argc < 2) {
		cout << "Please specify input image path" << endl;
		return 0;
	}
	help();

	Mat img = imread(argv[1]);

	//if fail to read the image
	if (img.empty())
	{
		cout << "Error loading the image: "<<argv[1] << endl;
		return -1;
	}
	
	//Create a window
	namedWindow("Window", 1);
	Mat imgShown;

	bool showResult = false;
	while (1) {
		if (!showResult) {
			img.copyTo(imgShown);
			//set the callback function for any mouse event
			setMouseCallback("Window", CallBackFunc, NULL);

			if (sQuadPoints.size() >0 && sQuadPoints.size() < 4) {
				line(imgShown, sCurrentPt, sQuadPoints[0], Scalar(255, 0, 0), 1, 4);
				line(imgShown, sCurrentPt, sQuadPoints.back(), Scalar(255, 0, 0), 1, 4);

				polylines(imgShown, sQuadPoints, false, Scalar(250, 255, 0), 1);
			}
			else if(sQuadPoints.size()==4)
				polylines(imgShown, sQuadPoints, true, Scalar(250, 250, 0), 1);
		}
		//show the image
		
		imshow("Window", imgShown);


		// Wait until user press some key
		int key = waitKey(10);
		if (key == 27)
			break;
		else if (key == 13) {
			if (sQuadPoints.size() == Quad) {
				//Get bound box
				Rect2i rect = boundingRect(sQuadPoints);
				vector<Point2i> dstPts(4);
				dstPts[0] = Point2i(0, 0);
				dstPts[1] = Point2i(0, rect.height);
				dstPts[2] = Point2i(rect.width, rect.height);
				dstPts[3] = Point2i(rect.width, 0);
				//Do homograph
				Mat H = findHomography(sQuadPoints, dstPts);
				std::cout << "Homography Matrix is\n"<< H << std::endl;
				//store the result in @imgShown
				imgShown.create(rect.height,rect.width, img.type());
				warpPerspective(img, imgShown, H, imgShown.size());
				showResult = true;
			}
		}
	}

	return 0;
}