
#include"header.h"
#include <opencv2/features2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void detectAndMatchFeatures(int argc, char** argv)
{
	if (argc < 3) {
		std::cout << "Pleae specify two images!" << std::endl;
		return ;
	}
	Mat img0 = imread(argv[1]);
	Mat img1 = imread(argv[2]);
	if (img0.empty() || img1.empty()) {
		std::cout << "Fail to load the two images!" << std::endl;
		return;
	}

	//1. detect features with FAST
	//FastFeatureDetector fast;
	Ptr<SURF> surf = SURF::create();
	vector<KeyPoint> keyPts0, keyPts1;
	surf->detect(img0, keyPts0);  
	surf->detect(img1, keyPts1);  

	cout << "img0--number of keypoints: " << keyPts0.size() << endl;
	cout << "img1--number of keypoints: " << keyPts1.size() << endl;

	//2. detect and compute feature descriptor
	Mat desc0, desc1;
	surf->compute(img0, keyPts0, desc0); //detectAndCompute(img0, Mat(), keyPts0, desc0);
	surf->compute(img1, keyPts1, desc1); //detectAndCompute(img1, Mat(), keyPts1, desc1);
	
	//3. matching
	//BruteForceMatcher< L2<float> > matcher; //FlannBasedMatcher matcher;  
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(desc0, desc1, matches);
	cout << "number of matches: " << matches.size() << endl;

	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < desc0.rows; i++){
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < desc0.rows; i++){
		if (matches[i].distance <= max(2 * min_dist, 0.02)){
			good_matches.push_back(matches[i]);
		}
	}
	//4. show result
	Mat matchImg;
	drawMatches(img0, keyPts0, img1, keyPts1, good_matches, matchImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matching result", matchImg);
	imwrite("match_result.png", matchImg);
	waitKey(0);
}