#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

using namespace cv;

float aggregated_cost(int rows, int cols, Vec3f& plane, int& window_radius,
                      float& gamma, float& alpha, float& trunc_col, float& trunc_grad)
{

}


int main() {
    Mat imgL = imread("/home/henry/project/data/cones/im2.png");
    Mat imgR = imread("/home/henry/project/data/cones/im6.png");

    int max_disparity = 64;
    int cols = imgL.cols;
    int rows = imgL.rows;
    Mat depthMap = Mat::zeros(rows, cols, CV_32FC3);

    //step1: random initialization
    srand (time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int rand_depth = rand() % (max_disparity + 1);
            float rand_normal_x = rand() / float(RAND_MAX);
            float rand_normal_y = rand() / float(RAND_MAX);
            float rand_normal_z = rand() / float(RAND_MAX);
            Vec3f& plane = depthMap.at<Vec3f>(i, j);
            plane[0] = -rand_normal_x / rand_normal_z;
            plane[1] = -rand_normal_y / rand_normal_z;
            plane[2] = (rand_normal_x * j + rand_normal_y * i + rand_normal_z * rand_depth) / rand_normal_z;
        }
    }

    //step2: iteration
    int max_iteration = 4;
    int window_radius = 17;
    int neighbor_radius = 17;
    float gamma = 10.0f;
    float alpha = 0.9f;
    float trunc_col = 10.0f;
    float trunc_grad = 2.0f;

    for(int iter_idx = 0; iter_idx < max_iteration; ++iter_idx)
    {
        if(0 == iter_idx % 2)
        {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    //step2.1: spatial iteration
                    for (int dy = 0; dy < neighbor_radius; ++dy) {
                        for (int dx = -neighbor_radius; dx < neighbor_radius; ++dx) {
                            
                        }
                    }


                    //step2.2: view iteration

                    //step2.3: plane refinement
                }
            }
        }


    }

    return 0;
}