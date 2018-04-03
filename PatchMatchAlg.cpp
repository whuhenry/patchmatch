//
// Created by YanZhao on 2018/4/2.
//

#include "PatchMatchAlg.h"
#include <random>

using namespace cv;

PatchMatchAlg::PatchMatchAlg()  : random(time(nullptr)) {
    gamma_ = 0.9;
    alpha_ = 10;
    trunc_col_ = 10;
    trunc_grad_ = 2;
    max_dissimilarity = (1 - alpha_) * trunc_col_ + alpha_ * trunc_grad_;
    window_radius_ = 17;
    neighbor_radius_ = 17;
    max_disparity_ = 64;
    window_pixel_count_ = (window_radius_ * 2 + 1) * (window_radius_ * 2 + 1);
}


void PatchMatchAlg::random_init(std::shared_ptr<Image> img) {
    //step1: random initialization
    img->plane_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    img->cost_mat_ = Mat::zeros(rows_, cols_, CV_32F);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            int rand_depth = random() % (max_disparity_ + 1);
            float rand_normal_x = random() / float(random.max());
            float rand_normal_y = random() / float(random.max());
            float rand_normal_z = random() / float(random.max());
            Vec3f normal(rand_normal_x, rand_normal_y, rand_normal_z);
            normal = normal / norm(normal);
            img->plane_mat_.at<Vec3f>(i, j) = Image::normal_to_plane(j, i, rand_depth, normal);
        }
    }

    //initial gradient map
    img->grad_mat_ = Mat::zeros(rows_, cols_, CV_32FC3);
    for (int i = 1; i < rows_; ++i) {
        for (int j = 1; j < cols_; ++j) {
            img->grad_mat_.at<Vec3f>(i, j)
                    = img->image_mat_.at<Vec3b>(i - 1, j)
                            + img->image_mat_.at<Vec3b>(i, j - 1) - 2 * img->image_mat_.at<Vec3b>(i, j);
        }
    }
}

void PatchMatchAlg::solve(std::shared_ptr<Image> imgL, std::shared_ptr<Image> imgR) {
    imgL_ = imgL;
    imgR_ = imgR;
    assert(imgL->rows_ == imgR->rows_);
    assert(imgL->cols_ == imgR->cols_);
    rows_ = imgL_->rows_;
    cols_ = imgR_->cols_;

    //random initialization
    random_init(imgL_);
    random_init(imgR_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            imgL_->cost_mat_.at<float>(i, j) = aggregated_cost(imgL_, imgR_, i, j, imgL_->plane_mat_.at<Vec3f>(i, j), L2R);
            imgR_->cost_mat_.at<float>(i, j) = aggregated_cost(imgR_, imgL_, i, j, imgR_->plane_mat_.at<Vec3f>(i, j), R2L);
        }
    }

    //iteration solve
    for (int iteration = 0; iteration < 3; ++iteration) {
        //Left to Right
        if (0 == iteration % 2) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    spatial_match(imgL_, imgR_, j, i, true, L2R);
                    view_match(j, i, L2R);
                    plane_refine(imgL_, imgR_, j, i, L2R);
                }
            }
        } else {
            for (int i = rows_ - 1; i >= 0; --i) {
                for (int j = cols_ - 1; j >= 0; --j) {
                    spatial_match(imgL_, imgR_, j, i, false, L2R);
                    view_match(j, i, L2R);
                    plane_refine(imgL_, imgR_, j, i, L2R);
                }
            }
        }

        //Right to Left
        if (0 == iteration % 2) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    spatial_match(imgR_, imgL_, j, i, true, R2L);
                    view_match(j, i, L2R);
                    plane_refine(imgR_, imgL_, j, i, R2L);
                }
            }
        } else {
            for (int i = rows_ - 1; i >= 0; --i) {
                for (int j = cols_ - 1; j >= 0; --j) {
                    spatial_match(imgR_, imgL_, j, i, false, R2L);
                    view_match(j, i, L2R);
                    plane_refine(imgR_, imgL_, j, i, R2L);
                }
            }
        }
    }

    //TODO: post process

    //show result
    show_result();
}

void PatchMatchAlg::spatial_match(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                                  int x, int y, bool ul_to_br, MatchDirection direction) {
    Vec3f &plane_center = img1->plane_mat_.at<Vec3f>(y, x);
    if (ul_to_br) {
        for (int dy = -neighbor_radius_; dy < neighbor_radius_; ++dy) {
            if (y + dy < 0) {
                continue;
            }
            if (y + dy >= img1->rows_) {
                break;
            }
            for (int dx = -neighbor_radius_; dx < neighbor_radius_; ++dx) {
                if (x + dx < 0) {
                    continue;
                }
                if (x + dx >= img1->cols_ || (x >= 0 && y >= 0)) {
                    break;
                }
                Vec3f &plane_comp = img1->plane_mat_.at<Vec3f>(y + dy, x + dx);
                float new_cost = aggregated_cost(img1, img2, y, x, plane_comp, direction);
                if (img1->cost_mat_.at<float>(y, x) > new_cost) {
                    img1->cost_mat_.at<float>(y, x) = new_cost;
                    plane_center = plane_comp;
                }
            }
        }
    } else {
        //bottom right pixel
        for (int dy = 1; dy <= neighbor_radius_; ++dy) {
            if (y + dy < 0) {
                continue;
            }
            if (y + dy >= img1->rows_) {
                break;
            }
            for (int dx = 1; dx <= neighbor_radius_; ++dx) {
                if (x + dx < 0) {
                    continue;
                }
                if (x + dx >= img1->cols_ || (x >= 0 && y >= 0)) {
                    break;
                }

                Vec3f &plane_comp = img1->plane_mat_.at<Vec3f>(y + dy, x + dx);
                float new_cost = aggregated_cost(img1, img2, y, x, plane_comp, direction);
                if (img1->cost_mat_.at<float>(y, x) > new_cost) {
                    img1->cost_mat_.at<float>(y, x) = new_cost;
                    plane_center = plane_comp;
                }
            }
        }
    }
}

void PatchMatchAlg::view_match(int x, int y, MatchDirection direction) {
    if(L2R == direction){
        //view match from left to right
        Vec3f &plane_center = imgL_->plane_mat_.at<Vec3f>(y, x);
        for (int right_x = max(x - max_disparity_, 0); right_x < x; ++right_x) {
            Vec3f &plane_comp = imgR_->plane_mat_.at<Vec3f>(y, right_x);
            float left_x_continue = disp_from_plane(right_x, y, plane_comp);
            int left_x = lround(left_x_continue);
            if (left_x == x) {
                float new_cost = aggregated_cost(imgL_, imgR_, y, x, plane_comp, L2R);
                if (imgL_->cost_mat_.at<float>(y, x) > new_cost) {
                    imgL_->cost_mat_.at<float>(y, x) = new_cost;
                    plane_center = plane_comp;
                }
            }
        }
    } else {
        //view match from right to left
        Vec3f &plane_center = imgR_->plane_mat_.at<Vec3f>(y, x);
        for (int left_x = x; left_x < min(x + max_disparity_, imgL_->cols_); ++left_x) {
            Vec3f &plane_comp = imgL_->plane_mat_.at<Vec3f>(y, left_x);
            float right_x_continue = disp_from_plane(left_x, y, plane_comp);
            int right_x = lround(right_x_continue);
            if (right_x == x ) {
                float new_cost = aggregated_cost(imgR_, imgL_, y, x, plane_comp, R2L);
                if (imgR_->cost_mat_.at<float>(y, x) > new_cost) {
                    imgR_->cost_mat_.at<float>(y, x) = new_cost;
                    plane_center = plane_comp;
                }
            }
        }
    }

}

void PatchMatchAlg::plane_refine(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                                 int x, int y, MatchDirection direction) {
    float delta_z_max = max_disparity_ / 2.0f;
    float delta_n_max = 1;
    Vec3f &plane_center = img1->plane_mat_.at<Vec3f>(y, x);
    Vec3f normal = Image::plane_to_normal(plane_center);
    float disp = disp_from_plane(x, y, plane_center);
    Vec3f plane_comp;
    float pre_cost = img1->cost_mat_.at<float>(y, x);
    while (delta_z_max >= 0.1) {
        Vec3f delta_normal;
        float delta_z = (random() / float(random.max()) - 0.5f) * 2 * delta_z_max;
        delta_normal[0] = (random() / float(random.max()) - 0.5f) * 2 * delta_n_max;
        delta_normal[1] = (random() / float(random.max()) - 0.5f) * 2 * delta_n_max;
        delta_normal[2] = (random() / float(random.max()) - 0.5f) * 2 * delta_n_max;
        disp += delta_z;
        delta_normal /= norm(delta_normal);
        normal += delta_normal;
        normal /= norm(normal);
        plane_comp = Image::normal_to_plane(x, y, disp, normal);
        float new_cost = aggregated_cost(img1, img2, y, x, plane_comp, direction);
        if (new_cost < pre_cost) {
            pre_cost = new_cost;
            plane_center = plane_comp;
        }

        delta_z_max /= 2.0f;
        delta_n_max /= 2.0f;
    }
    img1->cost_mat_.at<float>(y, x) = pre_cost;
}

float PatchMatchAlg::aggregated_cost(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                                     int row, int col, cv::Vec3f &plane, MatchDirection direction) {
    if (row < window_radius_ || col < window_radius_ || row + window_radius_ >= rows_ ||
        col + window_radius_ >= cols_) {
        return float(max_disparity_ * window_pixel_count_);
    }

    float cost = 0;
    for (int dx = -window_radius_; dx <= window_radius_; ++dx) {
        for (int dy = -window_radius_; dy <= window_radius_; ++dy) {
            Vec3b &Ip = img1->image_mat_.at<Vec3b>(row, col);
            Vec3b &Iq = img1->image_mat_.at<Vec3b>(row + dy, col + dx);
            Vec3f &Gq = img1->grad_mat_.at<Vec3f>(row + dy, col + dx);
            float weight = exp(-l1_norm(Ip, Iq) / gamma_);

            float corresponding_x = col + dx - direction * disp_from_plane(col + dx, row + dy, plane);
            if (0 <= corresponding_x && col - 1 >= corresponding_x) {
                Vec3f iq_corresponding = img2->get_pixel_bilinear(corresponding_x, row + dy);
                Vec3f gq_corresponding = img2->get_grad_bilinear(corresponding_x, row + dy);
                float dissimilarity = (1 - alpha_) * min(l1_norm(Iq, iq_corresponding), trunc_col_)
                                      + alpha_ * min(l1_norm(Gq, gq_corresponding), trunc_grad_);

                cost += weight * dissimilarity;
            } else {
                cost += weight * max_dissimilarity;
            }
        }
    }

    return cost;
}

inline float PatchMatchAlg::disp_from_plane(int x, int y, cv::Vec3f &plane) {
    return x * plane[0] + y * plane[1] + plane[2];
}

template<class T1, class T2>
inline float PatchMatchAlg::l1_norm(cv::Vec<T1, 3> v1, cv::Vec<T2, 3> v2) {
    float result = 0;
    for (int i = 0; i < 3; ++i) {
        result += abs(v1[i] - v2[i]);
    }

    return result;
}

void PatchMatchAlg::show_result() {
    cv::Mat disp_mat = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat.at<unsigned char>(i, j)
                    = (unsigned char)(disp_from_plane(j, i, imgL_->plane_mat_.at<Vec3f>(i, j)) / max_disparity_ * 255);
        }
    }
    imwrite(R"(/mnt/f/Data/Benchmark/teddy/disp.bmp)", disp_mat);
}
