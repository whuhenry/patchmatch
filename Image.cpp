//
// Created by henry on 18-2-1.
//

#include "image.h"


using namespace cv;

Image::Image() : random(time(nullptr)) {
    max_disparity_ = 64;
}

void Image::load(std::string image_path) {
    image_mat_ = imread(image_path);
    rows_ = image_mat_.rows;
    cols_ = image_mat_.cols;
}

void Image::init(float alpha, float gamma, float trunc_col, float trunc_grad) {
    alpha_ = alpha;
    gamma_ = gamma;
    trunc_col_ = trunc_col;
    trunc_grad_ = trunc_grad;
    max_dissimilarity = (1 - alpha_) * trunc_col_ + alpha_ * trunc_grad_;

    //step1: random initialization
    plane_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            int rand_depth = random() % (max_disparity_ + 1);
            float rand_normal_x = random() / float(random.max());
            float rand_normal_y = random() / float(random.max());
            float rand_normal_z = random() / float(random.max());
            Vec3f normal(rand_normal_x, rand_normal_y, rand_normal_z);
            normal = normal / norm(normal);
            plane_mat_.at<Vec3f>(i, j) = normal_to_plane(j, i, rand_depth, normal);
        }
    }

    //initial gradient map
    grad_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    for (int i = 1; i < rows_; ++i) {
        for (int j = 1; j < cols_; ++j) {
            grad_mat_.at<Vec3f>(i, j)
                    = image_mat_.at<Vec3b>(i - 1, j) + image_mat_.at<Vec3b>(i, j - 1) - 2 * image_mat_.at<Vec3b>(i, j);
        }
    }
}

cv::Vec3f Image::get_pixel_bilinear(Mat &mat, float x, float y) {
    auto ix = int(x);
    auto iy = int(y);
    float u = y - iy;
    float v = x - ix;
    return (1 - u) * (1 - v) * mat.at<Vec3f>(iy, ix) + (1 - u) * v * mat.at<Vec3f>(iy, ix + 1)
           + u * (1 - v) * mat.at<Vec3f>(iy + 1, ix) + u * v * mat.at<Vec3f>(iy + 1, ix + 1);
}

Vec3f Image::normal_to_plane(int x, int y, float z, cv::Vec3f normal) {
    Vec3f plane;
    plane[0] = -normal[0] / normal[2];
    plane[1] = -normal[1] / normal[2];
    plane[2] = (normal[0] * x + normal[1] * y + normal[2] * z) / normal[2];
    return plane;
}

Vec3f Image::plane_to_normal(cv::Vec3f plane) {
    Vec3f normal;
    normal[2] = sqrt(1 / (plane[0] * plane[0] + plane[1] * plane[1] + 1));
    normal[0] = -plane[0] * normal[2];
    normal[1] = -plane[1] * normal[2];
    return normal;
}