//
// Created by henry on 18-2-1.
//

#ifndef PATCHMATCH_IMAGE_H
#define PATCHMATCH_IMAGE_H

#include <string>
#include <opencv2/opencv.hpp>

class Image {
public:
    Image();
    ~Image();
    void load(std::string image_path);
    void init(float alpha, float gamma, float trunc_col, float trunc_grad);
	void match();
	void spatial_match();

    static cv::Vec3f get_pixel_billinear(cv::Mat& mat, float x, float y);
    static float l1_norm(cv::Vec3f v1, cv::Vec3f v2);

    cv::Mat image_mat_, plane_mat_, grad_mat_;
    Image* match_image;

private:
    float gamma_, alpha_, trunc_col_, trunc_grad_, max_dissimilarity;
    int window_radius_, neighbor_radius_, max_disparity_;
    int rows_, cols_;
    float aggregated_cost(int row, int col, cv::Vec3f& plane);
};


#endif //PATCHMATCH_IMAGE_H
