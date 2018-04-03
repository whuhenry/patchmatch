//
// Created by henry on 18-2-1.
//

#include "image.h"


using namespace cv;

Image::Image() {
}

void Image::load(std::string image_path) {
    image_mat_ = imread(image_path);
    rows_ = image_mat_.rows;
    cols_ = image_mat_.cols;
}