//
// Created by henry on 18-2-1.
//

#ifndef PATCHMATCH_IMAGE_H
#define PATCHMATCH_IMAGE_H

#include <string>
#include <random>
#include <opencv2/opencv.hpp>


class Image {
public:
    Image();
    ~Image() = default;
    void load(std::string);

    inline cv::Vec3f get_pixel_bilinear(float x, float y){
        auto ix = int(x);
        auto iy = int(y);
        float u = y - iy;
        float v = x - ix;
        if (iy == y && ix == x) {
            return image_mat_.at<cv::Vec3b>(iy, ix);
        } else if (iy == y && ix != x) {
            return (1 - v) * image_mat_.at<cv::Vec3b>(iy, ix) + v * image_mat_.at<cv::Vec3b>(iy, ix + 1);
        } else if (iy != y && ix == x) {
            return (1 - u) * image_mat_.at<cv::Vec3b>(iy, ix) + u * image_mat_.at<cv::Vec3b>(iy + 1, ix);
        } else {
            return (1 - u) * (1 - v) * image_mat_.at<cv::Vec3b>(iy, ix) + (1 - u) * v * image_mat_.at<cv::Vec3b>(iy, ix + 1)
                   + u * (1 - v) * image_mat_.at<cv::Vec3b>(iy + 1, ix) + u * v * image_mat_.at<cv::Vec3b>(iy + 1, ix + 1);
        }
    }
    inline cv::Vec3f get_grad_bilinear(float x, float y){
        auto ix = int(x);
        auto iy = int(y);
        float u = y - iy;
        float v = x - ix;
        if (iy == y && ix == x) {
            return grad_mat_.at<cv::Vec3f>(iy, ix);
        } else if (iy == y && ix != x) {
            return (1 - v) * grad_mat_.at<cv::Vec3f>(iy, ix) + v * grad_mat_.at<cv::Vec3f>(iy, ix + 1);
        } else if (iy != y && ix == x) {
            return (1 - u) * grad_mat_.at<cv::Vec3f>(iy, ix) + u * grad_mat_.at<cv::Vec3f>(iy + 1, ix);
        } else {
            return (1 - u) * (1 - v) * grad_mat_.at<cv::Vec3f>(iy, ix) + (1 - u) * v * grad_mat_.at<cv::Vec3f>(iy, ix + 1)
                   + u * (1 - v) * grad_mat_.at<cv::Vec3f>(iy + 1, ix) + u * v * grad_mat_.at<cv::Vec3f>(iy + 1, ix + 1);
        }
    }
    static inline cv::Vec3f normal_to_plane(int x, int y, float z, cv::Vec3f normal) {
        cv::Vec3f plane;
        plane[0] = -normal[0] / normal[2];
        plane[1] = -normal[1] / normal[2];
        plane[2] = (normal[0] * x + normal[1] * y + normal[2] * z) / normal[2];
        return plane;
    }
    static inline cv::Vec3f plane_to_normal(cv::Vec3f plane){
        cv::Vec3f normal;
        normal[2] = sqrt(1 / (plane[0] * plane[0] + plane[1] * plane[1] + 1));
        normal[0] = -plane[0] * normal[2];
        normal[1] = -plane[1] * normal[2];
        return normal;
    }

    cv::Mat image_mat_, plane_mat_, grad_mat_, cost_mat_;
    int rows_, cols_;
};


#endif //PATCHMATCH_IMAGE_H
