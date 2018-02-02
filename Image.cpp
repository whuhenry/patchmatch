//
// Created by henry on 18-2-1.
//

#include "Image.h"

using namespace cv;

Image::Image()
{

}

Image::~Image()
{

}

void Image::load(std::string image_path)
{
    image_mat_ = imread(image_path);
    rows_ = image_mat_.rows;
    cols_ = image_mat_.cols;
}

void Image::init(float alpha, float gamma, float trunc_col, float trunc_grad)
{
    alpha_ = alpha;
    gamma_ = gamma;
    trunc_col_ = trunc_col;
    trunc_grad_ = trunc_grad;
    max_dissimilarity = (1 - alpha_) * trunc_col_ + alpha_ * trunc_grad_;

    //step1: random initialization
    srand (time(NULL));
    plane_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            int rand_depth = rand() % (max_disparity_ + 1);
            float rand_normal_x = rand() / float(RAND_MAX);
            float rand_normal_y = rand() / float(RAND_MAX);
            float rand_normal_z = rand() / float(RAND_MAX);
            Vec3f& plane = plane_mat_.at<Vec3f>(i, j);
            plane[0] = -rand_normal_x / rand_normal_z;
            plane[1] = -rand_normal_y / rand_normal_z;
            plane[2] = (rand_normal_x * j + rand_normal_y * i + rand_normal_z * rand_depth) / rand_normal_z;
        }
    }

    //initial gradient map
    grad_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    for (int i = 1; i < rows_; ++i) {
        for (int j = 1; j < cols_; ++j) {
            grad_mat_.at<Vec3f>(i, j)
                    = image_mat_.at<Vec3f>(i - 1, j) + image_mat_.at<Vec3f>(i, j - 1) - 2 * image_mat_.at<Vec3f>(i, j);
        }
    }
}

float Image::aggregated_cost(int row, int col, Vec3f& plane)
{
    if(row < window_radius_ || col < window_radius_ || row + window_radius_ >= rows_ || col + window_radius_ >= cols_)
    {
        return float(max_disparity_ * 1225);
    }

    float cost = 0;
    for (int dx = -window_radius_; dx <= window_radius_; ++dx) {
        for (int dy = -window_radius_; dy <= window_radius_; ++dy) {
            Vec3f& Ip = image_mat_.at<Vec3f>(row, col);
            Vec3f& Iq = image_mat_.at<Vec3f>(row + dy, col + dx);
            Vec3f& Gq = grad_mat_.at<Vec3f>(row + dy, col + dx);
            float l1_norm0 = 0.0f;
            for (int i = 0; i < 3; ++i) {
                l1_norm0 += abs(Ip[i] - Iq[i]);
            }
            float weight = exp(-l1_norm0 / gamma_);

            float disparity_q = plane.dot(Vec3f(row + dy, col + dx, 1));
            float corresponding_x = col + dx - disparity_q;
            if(0 <= corresponding_x && col - 1 > corresponding_x)
            {
                Vec3f iq_corresponding = get_pixel_billinear(match_image->image_mat_, corresponding_x, row + dy);
                Vec3f gq_corresponding = get_pixel_billinear(match_image->grad_mat_, corresponding_x, row + dy);
                float dissimilarity = (1 - alpha_) * min(l1_norm(Iq, iq_corresponding), trunc_col_)
                                      + alpha_ * min(l1_norm(Gq, gq_corresponding), trunc_grad_);

                cost += weight * dissimilarity;
            }
            else
            {
                cost += weight * max_dissimilarity;
            }
        }
    }

	return cost;
}

cv::Vec3f Image::get_pixel_billinear(Mat& mat, float x, float y)
{
    int ix = int(x);
    int iy = int(y);
    float u = y - iy;
    float v = x - ix;
    return (1 - u) * (1 - v) * mat.at<Vec3f>(iy, ix) + (1 - u) * v * mat.at<Vec3f>(iy, ix + 1)
           + u * (1 - v) * mat.at<Vec3f>(iy + 1, ix) + u * v * mat.at<Vec3f>(iy + 1, ix + 1);
}

float Image::l1_norm(Vec3f v1, Vec3f v2)
{
    float result = 0;
    for (int i = 0; i < 3; ++i) {
        result += abs(v1[i] - v2[i]);
    }

    return result;
}
