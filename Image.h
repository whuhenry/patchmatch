//
// Created by henry on 18-2-1.
//

#ifndef PATCHMATCH_IMAGE_H
#define PATCHMATCH_IMAGE_H

#include <string>
#include <opencv2/opencv.hpp>


class Image{
public:
    Image();
    ~Image();
    void load(std::string);

    inline void get_pixel_bilinear(float x, float y, float* out_pixel){
        int ix = int(x);
        int iy = int(y);
        int center = (iy * cols_ + ix) * 3;
        float u = y - iy;
        float v = x - ix;
        if (iy == y && ix == x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = image_[center + i];
            }
        } else if (iy == y && ix != x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - v) * image_[center + i] + v * image_[center + i + 3];
            }
        } else if (iy != y && ix == x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - u) * image_[center + i] + u * image_[center + i + 3 * cols_];
            }
        } else {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - u) * (1 - v) * image_[center + i] + (1 - u) * v * image_[center + i + 3]
                        + u * (1 - v) * image_[center + i + 3 * cols_] + u * v * image_[center + i + 3 * cols_ + 3];
            }
        }
    }

    inline void get_grad_bilinear(float x, float y, float* out_pixel){
        int ix = int(x);
        int iy = int(y);
        int center = (iy * cols_ + ix) * 3;
        float u = y - iy;
        float v = x - ix;
        if (iy == y && ix == x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = grad_[center + i];
            }
        } else if (iy == y && ix != x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - v) * grad_[center + i] + v * grad_[center + i + 3];
            }
        } else if (iy != y && ix == x) {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - u) * grad_[center + i] + u * grad_[center + i + 3 * cols_];
            }
        } else {
            for (int i = 0; i < 3; i++) {
                out_pixel[i] = (1 - u) * (1 - v) * grad_[center + i] + (1 - u) * v * grad_[center + i + 3]
                               + u * (1 - v) * grad_[center + i + 3 * cols_] + u * v * grad_[center + i + 3 * cols_ + 3];
            }
        }
    }
    static inline void normal_to_plane(int x, int y, float z, float* normal, float *plane) {
        plane[0] = -normal[0] / normal[2];
        plane[1] = -normal[1] / normal[2];
        plane[2] = (normal[0] * x + normal[1] * y + normal[2] * z) / normal[2];
    }

    static void show_disp(float* plane, int row, int col, int max_disp);

    //cv::Mat image_mat_, plane_mat_, grad_mat_, cost_mat_;
    float *image_;
    float *plane_, *cost_, *normal_, *grad_;
    float *d_image_, *d_plane_, *d_cost_, *d_normal_, *d_grad_;
    int rows_, cols_;
};


#endif //PATCHMATCH_IMAGE_H
