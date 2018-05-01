//
// Created by henry on 18-2-1.
//

#include "Image.h"


using namespace cv;

Image::Image() {
    image_ = nullptr;
    grad_ = nullptr;
    plane_ = nullptr;
    cost_ = nullptr;
}

Image::~Image() {
    delete[] image_;
    image_ = nullptr;
    delete[] grad_;
    grad_ = nullptr;
    delete[] plane_;
    plane_ = nullptr;
    delete[] cost_;
    cost_ = nullptr;
    delete[] normal_;
    normal_ = nullptr;
}

void Image::load(std::string image_path) {
    cv::Mat image_mat = imread(image_path);
    rows_ = image_mat.rows;
    cols_ = image_mat.cols;

    image_ = new float[rows_ * cols_ * 3];
    grad_ = new float[rows_ * cols_ * 3];
    plane_ = new float[rows_ * cols_ * 3];
    cost_ = new float[rows_ * cols_];
    normal_ = new float[rows_ * cols_ * 3];
    Mat float_mat;
    image_mat.convertTo(float_mat, CV_32F, 1 / 255.0f);
    memcpy(image_, image_mat.data, rows_ * cols_ * 3 * sizeof(uint8_t));
    memset(grad_, 0, rows_ * cols_ * 3 * sizeof(short));

    for (int i = 0; i < rows_ - 1; ++i) {
        for (int j = 0; j < cols_ - 1; ++j) {
            for (int k = 0; k < 3; ++k) {
                int center_loc = (i * cols_ + j) * 3 + k;
                grad_[center_loc] = image_[center_loc + 3 * cols_] + image_[center_loc + 3] - image_[center_loc] * 2;
            }
        }
    }
}