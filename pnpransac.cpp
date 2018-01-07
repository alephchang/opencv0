#include<fstream>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/calib3d.hpp>
#include<vector>
using cv::Mat;
int pnpransac()
{
	std::ifstream fin("pnp_param.txt", std::ios::in);
	if (fin.bad()) {
		std::cout << "cannot load file" << std::endl;
	}
	int match_num;
	fin >> match_num;
	std::cout << match_num << std::endl;
	std::vector<cv::Point3f> pt3(match_num);
	std::vector<cv::Point2f> pt2(match_num);
	for (int i = 0; i < match_num; ++i) {
		fin >> pt3[i].x >> pt3[i].y >> pt3[i].z >> pt2[i].x >> pt2[i].y;
	}
	Mat K = (cv::Mat_<float>(3, 3) << 517.3, 0, 325.1,
		0, 516.5, 249.7,
		0, 0, 1.0);
	Mat rvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
	Mat tvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
	Mat inliers;
	cv::solvePnPRansac(pt3, pt2, K, Mat(), rvec, tvec, true, 100, 4.0f, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
	Mat rvec1, tvec1;
	cv::solvePnPRansac(pt3, pt2, K, Mat(), rvec1, tvec1, false, 100, 4.0f, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
	std::cout << "roation by pnp with useExtrinsicGuess=true : " << rvec << std::endl;
	std::cout << "roation by pnp with useExtrinsicGuess=false: " << rvec1 << std::endl;
	return 0;
}