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
    normal_ = nullptr;
}

Image::~Image() {
    if(nullptr != image_) {
        delete[] image_;
        image_ = nullptr;
    }
    if(nullptr != grad_) {
        delete[] grad_;
        grad_ = nullptr;
    }
    if (nullptr != plane_) {
        delete[] plane_;
        plane_ = nullptr;
    }
    if (nullptr != plane_) {
        delete[] cost_;
        cost_ = nullptr;
    }
    if (nullptr != normal_) {
        delete[] normal_;
        normal_ = nullptr;
    }
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
    memcpy(image_, float_mat.data, rows_ * cols_ * 3 * sizeof(float));

    int rowoff = 3 * cols_, coloff = 3, center_loc = 0;
    for (int i = 0; i < rows_; ++i) {
        if (i == rows_ - 1) {
            rowoff = 0;
        }
        for (int j = 0; j < cols_; ++j) {
            if (j == cols_ - 1) {
                coloff = 0;
            }
            for (int k = 0; k < 3; ++k) {
                grad_[center_loc] = image_[center_loc + rowoff] + image_[center_loc + coloff] - image_[center_loc] * 2;
                ++center_loc;
            }
        }
    }
}

void Image::show_disp(float *plane, int row, int col, int max_disp) {
    cv::Mat disp_mat = Mat::zeros(row, col, CV_8U);
    int offset = 0;
    for (int y = 0; y < row; y++) {
        for (int x = 0; x < col; x++) {
            disp_mat.at<unsigned char>(y, x)
                    = (unsigned char) ((x * plane[offset] + y * plane[offset + 1] + plane[offset + 2]) / max_disp *
                                       255);
            offset += 3;
        }
    }
    imshow("disp", disp_mat);
    waitKey(0);
}
