#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]) {

#ifdef _WIN64
	Mat imgL = imread("F:\\Data\\Benchmark\\cones\\im2.png");
	Mat imgR = imread("F:\\Data\\Benchmark\\cones\\im6.png");
#else
	Mat imgL = imread("/home/henry/project/data/cones/im2.png");
	Mat imgR = imread("/home/henry/project/data/cones/im6.png");
#endif

	namedWindow("test");
	imshow("test", imgL);
	waitKey(0);

    return 0;
}