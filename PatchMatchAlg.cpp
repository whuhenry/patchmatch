//
// Created by YanZhao on 2018/4/2.
//

#include "PatchMatchAlg.h"
#include <random>
#include <boost/log/trivial.hpp>

using namespace cv;

PatchMatchAlg::PatchMatchAlg() {
    gamma_ = 10;
    alpha_ = 0.9;
    trunc_col_ = 10;
    trunc_grad_ = 2;
    max_dissimilarity = (1 - alpha_) * trunc_col_ + alpha_ * trunc_grad_;
    window_radius_ = 17;
    max_disparity_ = 64;
    window_pixel_count_ = (window_radius_ * 2 + 1) * (window_radius_ * 2 + 1);
}

PatchMatchAlg::~PatchMatchAlg() {
}


void PatchMatchAlg::random_init(Image* img) {
    //step1: random initialization

    long offset = 0;
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            float rand_depth = dis(random) * max_disparity_;
            if (rand_depth < 0) {
                rand_depth = -rand_depth;
            }
            float* normal = img->normal_ + offset;
            normal[0] = dis(random);
            normal[1] = dis(random);
            normal[2] = dis(random);
            l2_norm(normal);
            if(normal[2] < 0) {
                normal[2] = -normal[2];
            }
            Image::normal_to_plane(j, i, rand_depth, normal, img->plane_ + offset);
            offset += 3;
        }
    }
}

void PatchMatchAlg::solve(std::shared_ptr<Image> imgL, std::shared_ptr<Image> imgR) {
    BOOST_LOG_TRIVIAL(info) << "start";
    imgL_ = imgL.get();
    imgR_ = imgR.get();
    assert(imgL->rows_ == imgR->rows_);
    assert(imgL->cols_ == imgR->cols_);
    rows_ = imgL_->rows_;
    cols_ = imgL_->cols_;

    //random initialization
    random_init(imgL_);
    random_init(imgR_);
    int pixel_idx = 0;
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            imgL_->cost_[pixel_idx] = aggregated_cost(imgL_, imgR_, i, j, imgL_->plane_ + pixel_idx * 3, L2R);
            imgR_->cost_[pixel_idx] = aggregated_cost(imgR_, imgL_, i, j, imgR_->plane_ + pixel_idx * 3, R2L);
            ++pixel_idx;
        }
    }
    BOOST_LOG_TRIVIAL(info) << "init complete";


    //iteration solve
    for (int iteration = 0; iteration < 3; ++iteration) {

        spatial_match(iteration);

        view_match(iteration);

        plane_refine(iteration);

        BOOST_LOG_TRIVIAL(info) << "iteration " << iteration << " finished";
    }

    post_process();

    //show result
    show_result();
    //    write_result();
    BOOST_LOG_TRIVIAL(info) << "finish";
}

void PatchMatchAlg::save_disp_map(std::string path) {
    std::ofstream ofs;
    ofs.open(path);
    if(!ofs.is_open()) {
        return;
    }

    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex " << cols_ * rows_ << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "property float nx" << std::endl;
    ofs << "property float ny" << std::endl;
    ofs << "property float nz" << std::endl;
    ofs << "property uchar diffuse_red" << std::endl;
    ofs << "property uchar diffuse_green" << std::endl;
    ofs << "property uchar diffuse_blue" << std::endl;
    ofs << "end_header" << std::endl;

    int offset = 0;
    float disp, *normal;
    uint8_t *diffuse;
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            disp = disp_from_plane(j, i, imgL_->plane_ + offset);
            normal = imgL_->normal_ + offset;
            diffuse = imgL_->image_ + offset;
            ofs << j << " " << i << " " << disp << " "
                << normal[0] << " " << normal[1] << " " << normal[2] << " "
                << (int)diffuse[0] << " " << (int)diffuse[1] << " " << (int)diffuse[2] << std::endl;
            offset += 3;
        }
    }

    ofs.close();
}

void PatchMatchAlg::spatial_match(int iter_num) {
    int x_start, x_end, x_inc, y_start, y_end, y_inc;
    if (0 == iter_num % 2) {
        x_start = 0;
        x_end = cols_;
        x_inc = 1;
        y_start = 0;
        y_end = rows_;
        y_inc = 1;
    }
    else {
        x_start = cols_ - 1;
        x_end = -1;
        x_inc = -1;
        y_start = rows_ - 1;
        y_end = -1;
        y_inc = -1;
    }
    for (int view = 1; view >= 0; --view) {
        Image* base_img = view % 2 == 0 ? imgL_ : imgR_;
        Image* ref_img = view % 2 == 1 ? imgL_ : imgR_;
        MatchDirection direction = view % 2 == 0 ? L2R : R2L;
        for (int y = y_start; y != y_end; y += y_inc) {
            for (int x = x_start; x != x_end; x += x_inc) {
                float* plane_center = base_img->plane_ + ((y * cols_ + x) * 3);
                float* cost_center = base_img->cost_ + (y * cols_ + x);
                float* base_normal = base_img->normal_ + ((y * cols_ + x) * 3);

                //x_neighbor
                int x_neighbor = x - x_inc;
                if (x_neighbor >=0 && x_neighbor < cols_) {
                    float* plane_comp = base_img->plane_ + (y * cols_ + x_neighbor) * 3;
                    float new_cost = aggregated_cost(base_img, ref_img, y, x, plane_comp, direction, *cost_center);
                    if (new_cost < *cost_center) {
                        *cost_center = new_cost;
                        cpy_vec3(plane_center, plane_comp);
                        cpy_vec3(base_normal, base_img->normal_ + (y * cols_ + x_neighbor) * 3);
                    }
                }
                //y_neighbor
                int y_neighbor = y - y_inc;
                if (y_neighbor >=0 && y_neighbor < rows_) {
                    float* plane_comp = base_img->plane_ + (y_neighbor * cols_ + x) * 3;
                    float new_cost = aggregated_cost(base_img, ref_img, y, x, plane_comp, direction, *cost_center);
                    if (new_cost < *cost_center) {
                        *cost_center = new_cost;
                        cpy_vec3(plane_center, plane_comp);
                        cpy_vec3(base_normal, base_img->normal_ + (y_neighbor * cols_ + x) * 3);
                    }
                }
            }
        }
    }
}

void PatchMatchAlg::view_match(int iter_num) {
    int x_start, x_end, x_inc, y_start, y_end, y_inc;
    if (0 == iter_num % 2) {
        x_start = 0;
        x_end = cols_;
        x_inc = 1;
        y_start = 0;
        y_end = rows_;
        y_inc = 1;
    }
    else {
        x_start = cols_ - 1;
        x_end = -1;
        x_inc = -1;
        y_start = rows_ - 1;
        y_end = -1;
        y_inc = -1;
    }
    for (int view = 0; view < 2; ++view) {
        Image* base_img = view % 2 == 0 ? imgL_ : imgR_;
        Image* ref_img = view % 2 == 1 ? imgL_ : imgR_;
        MatchDirection direction = view % 2 == 0 ? L2R : R2L;
        for (int y = y_start; y != y_end; y += y_inc) {
            for (int x = x_start; x != x_end; x += x_inc) {
                float* ref_plane_center = ref_img->plane_ + ((y * cols_ + x) * 3);
                float disp = disp_from_plane(x, y, ref_plane_center);
                if (disp < 0.0f) {
                    disp = 0.0f;
                } else if (disp >= max_disparity_) {
                    disp = max_disparity_ - 1;
                }
                int correspond_x = lround(x + direction * disp);
                if (correspond_x < 0) {
                    correspond_x = 0;
                } else if (correspond_x >= cols_) {
                    correspond_x = cols_ - 1;
                }
                float* cost_center = base_img->cost_ + (y * cols_ + correspond_x);
                const float new_cost = aggregated_cost(base_img, ref_img, y, correspond_x, ref_plane_center, direction, *cost_center);
                if (new_cost < *cost_center) {
                    *cost_center = new_cost;
                    cpy_vec3(base_img->plane_ + (y * cols_ + correspond_x) * 3, ref_plane_center);
                    cpy_vec3(base_img->normal_ + (y * cols_ + correspond_x) * 3,
                             ref_img->normal_ + ((y * cols_) + x) * 3);
                }

            }
        }
    }
}

void PatchMatchAlg::plane_refine(int iter_num) {
    int x_start, x_end, x_inc, y_start, y_end, y_inc;
    if (0 == iter_num % 2) {
        x_start = 0;
        x_end = cols_;
        x_inc = 1;
        y_start = 0;
        y_end = rows_;
        y_inc = 1;
    }
    else {
        x_start = cols_ - 1;
        x_end = -1;
        x_inc = -1;
        y_start = rows_ - 1;
        y_end = -1;
        y_inc = -1;
    }
    float *normal, plane_comp[3], delta_normal[3], new_normal[3];
    float *plane_center, *cost_center, delta_z_max, delta_n_max, disp;
    for (int view = 0; view < 2; ++view) {
        Image* base_img = view % 2 == 0 ? imgL_ : imgR_;
        Image* ref_img = view % 2 == 1 ? imgL_ : imgR_;
        MatchDirection direction = view % 2 == 0 ? L2R : R2L;
        for (int y = y_start; y != y_end; y += y_inc) {
            for (int x = x_start; x != x_end; x += x_inc) {
                plane_center = base_img->plane_ + ((y * cols_ + x) * 3);
                cost_center = base_img->cost_ + (y * cols_ + x);
                delta_z_max = max_disparity_ / 2.0f;
                delta_n_max = 1;

                normal = base_img->normal_ + ((y * cols_ + x) * 3);
                float disp = disp_from_plane(x, y, plane_center);
                std::uniform_real_distribution<float> dis(-1.0, 1.0);
                while (delta_z_max >= 0.1) {
                    float delta_z = dis(random) * delta_z_max;
                    delta_normal[0] = dis(random) * delta_n_max;
                    delta_normal[1] = dis(random) * delta_n_max;
                    delta_normal[2] = dis(random) * delta_n_max;
                    float new_disp = disp + delta_z;
                    if (new_disp < 0) {
                        new_disp = -new_disp;
                    }
                    for (int i = 0; i < 3; ++i) {
                        new_normal[i] = normal[i] + delta_normal[i];
                    }
                    if (new_normal[2] < 0) {
                        new_normal[2] = -new_normal[2];
                    }
                    l2_norm(new_normal);
                    Image::normal_to_plane(x, y, new_disp, new_normal, plane_comp);
                    float new_cost = aggregated_cost(base_img, ref_img, y, x, plane_comp, direction, *cost_center);
                    if (new_cost < *cost_center) {
                        *cost_center = new_cost;
                        disp = new_disp;
                        cpy_vec3(plane_center, plane_comp);
                        cpy_vec3(normal, new_normal);
                    }

                    delta_z_max /= 2.0f;
                    delta_n_max /= 2.0f;
                }
            }
        }
    }
}

void PatchMatchAlg::post_process() {
    bool* valid_mask = new bool [rows_ * cols_];
    float* disp_l = new float [rows_ * cols_];
    for (int i = 0; i < rows_ * cols_; ++i) {
        valid_mask[i] = false;
    }
    for (int y = 0; y < rows_; ++y) {
        for (int x = 0; x < cols_; ++x) {
            float disparity_l = disp_from_plane(x, y, imgL_->plane_ + (y * cols_ + x) * 3);
            disp_l[y * cols_ + x] = disparity_l;
            if (disparity_l < 0 || disparity_l > max_disparity_) {
                continue;
            }
            int corresponding_x = (int)lround(x - disparity_l);
            if (corresponding_x < 0 || corresponding_x >= cols_) {
                continue;
            }
            float disparity_r = disp_from_plane(corresponding_x, y, imgR_->plane_ + (y * cols_ + corresponding_x) * 3);
            if (abs(disparity_r - disparity_l) <= 1) {
                valid_mask[y * cols_ + x] = true;
            }
        }
    }

    //fill disparity
    for (int y = 0; y < rows_; ++y) {
        for (int x = 0; x < cols_; ++x) {
            if (!valid_mask[y * cols_ + x]) {
                int search_x = x - 1;
                float search_disparity_l = -100000, search_disparity_r = -100000;
                while (search_x >= 0) {
                    if (valid_mask[y * cols_ + search_x]) {
                        search_disparity_l = disp_from_plane(x, y, imgL_->plane_ + (y * cols_ + search_x) * 3);
                        break;
                    }
                    --search_x;
                }
                search_x = x + 1;
                while (search_x < cols_) {
                    if (valid_mask[y * cols_ + search_x]) {
                        search_disparity_r = disp_from_plane(x, y, imgL_->plane_ + (y * cols_ + search_x) * 3);
                        break;
                    }
                    ++search_x;
                }
                disp_l[y * cols_ + x] = min(search_disparity_l, search_disparity_r);
            }
        }
    }

    //TODO: weighted median filter
    uint8_t *Ip, *Iq;
    for (int y = 0; y < rows_; ++y) {
        if (y < window_radius_ || y + window_radius_ >= rows_) {
            continue;
        }
        for (int x = 0; x < cols_; ++x) {
            if (valid_mask[y * cols_ + x] || x < window_radius_ || x + window_radius_ >= cols_) {
                continue;
            }
            Ip = imgL_->image_ + (y * cols_ + x) * 3;
            std::vector<std::pair<float, float>> weighted_disp;
            for (int dx = -window_radius_; dx <= window_radius_; ++dx) {
                for (int dy = -window_radius_; dy <= window_radius_; ++dy) {
                    int img_idx = ((y + dy) * cols_ + x + dx) * 3;
                    if (!valid_mask[img_idx / 3]) {
                        continue;
                    }
                    Iq = imgL_->image_ + img_idx;
                    float weight = exp(-l1_distance(Ip, Iq) / gamma_);
                    float disp = disp_from_plane(x + dx, y + dy, imgL_->plane_ + img_idx);
                    weighted_disp.emplace_back(disp, weight * disp);
                    std::sort(weighted_disp.begin(), weighted_disp.end(),
                              [](const std::pair<float, float>& p1, const std::pair<float, float>& p2) {
                                  return p1.second < p2.second;
                              });

                }
            }
        }
    }

    cv::Mat disp_mat = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat.at<unsigned char>(i, j)
                = (unsigned char)(disp_l[i * cols_ + j] / max_disparity_ * 255);
        }
    }
    imwrite(R"(/home/henry/disp_lr.bmp)", disp_mat);
    imshow("lr", disp_mat);
    waitKey(0);

    cv::Mat mask_mat = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (valid_mask[i * cols_ + j]) {
                mask_mat.at<unsigned char>(i, j) = 255;
            }
            else {
                mask_mat.at<unsigned char>(i, j) = 0;
            }
        }
    }
    imwrite(R"(/home/henry/mask.bmp)", mask_mat);

    delete[] valid_mask;
    valid_mask = nullptr;
    delete[] disp_l;
    disp_l = nullptr;
}

float PatchMatchAlg::aggregated_cost(Image* img1, Image* img2, int y, int x, float* plane,
                                     MatchDirection direction, float base_cost/* = FLT_MAX*/) {
    if (y < window_radius_ || x < window_radius_ || y + window_radius_ >= rows_ ||
        x + window_radius_ >= cols_) {
        return FLT_MAX;
    }

    //    float disp = disp_from_plane(x, y, plane);
    //    if (disp < 0 || disp > max_disparity_) {
    //        return FLT_MAX;
    //    }

    float cost = 0;
    uint8_t* Ip = img1->image_ + (y * cols_ + x) * 3;
    uint8_t* Iq = nullptr;
    short* Gq = nullptr;
    float iq_corresponding[3], gq_corresponding[3];
    for (int dy = -window_radius_; dy <= window_radius_; ++dy) {
        for (int dx = -window_radius_; dx <= window_radius_; ++dx) {
            Iq = img1->image_ + ((y + dy) * cols_ + x + dx) * 3;
            Gq = img1->grad_ + ((y + dy) * cols_ + x + dx) * 3;
            float weight = exp(-l1_distance(Ip, Iq) / gamma_);
            float disp = disp_from_plane(x + dx, y + dy, plane);
            if (disp < 0 || disp >= max_disparity_) {
                cost += weight * max_dissimilarity;
                continue;
            }
            float corresponding_x = x + dx - direction * disp;
            if (0 <= corresponding_x && cols_ - 1 >= corresponding_x) {
                img2->get_pixel_bilinear(corresponding_x, y + dy, iq_corresponding);
                img2->get_grad_bilinear(corresponding_x, y + dy, gq_corresponding);
                float dissimilarity = (1 - alpha_) * min(l1_distance(Iq, iq_corresponding), trunc_col_)
                    + alpha_ * min(l1_distance(Gq, gq_corresponding), trunc_grad_);
                cost += weight * dissimilarity;
            }
            else {
                cost += weight * max_dissimilarity;
            }
            if (cost > base_cost) {
                return FLT_MAX;
            }
        }
    }

    return cost;
}

inline float PatchMatchAlg::disp_from_plane(int x, int y, float* plane) {
    return x * plane[0] + y * plane[1] + plane[2];
}

template <class T1, class T2>
inline float PatchMatchAlg::l1_distance(T1* v1, T2* v2) {
    return abs(v1[0] - v2[0]) + abs(v1[1] - v2[1]) + abs(v1[2] - v2[2]);
}

void PatchMatchAlg::write_result() {
    cv::Mat disp_mat = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat.at<unsigned char>(i, j)
                = (unsigned char)(disp_from_plane(j, i, imgL_->plane_ + (i * cols_ + j) * 3) / max_disparity_ * 255);
        }
    }
    imwrite(R"(/home/henry/disp_l.bmp)", disp_mat);

    cv::Mat disp_mat_r = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat_r.at<unsigned char>(i, j)
                = (unsigned char)(disp_from_plane(j, i, imgR_->plane_ + (i * cols_ + j) * 3) / max_disparity_ * 255);
        }
    }
    imwrite(R"(/home/henry/disp_r.bmp)", disp_mat_r);
}

void PatchMatchAlg::show_result() {
    cv::Mat disp_mat = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat.at<unsigned char>(i, j)
                = (unsigned char)(disp_from_plane(j, i, imgL_->plane_ + (i * cols_ + j) * 3) / max_disparity_ * 255);
        }
    }
    cv::Mat disp_mat_r = Mat::zeros(rows_, cols_, CV_8U);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            disp_mat_r.at<unsigned char>(i, j)
                = (unsigned char)(disp_from_plane(j, i, imgR_->plane_ + (i * cols_ + j) * 3) / max_disparity_ * 255);
        }
    }
    imshow("left", disp_mat);
    imshow("right", disp_mat_r);
    waitKey(0);
}

template <class T>
inline void PatchMatchAlg::l2_norm(T* v1) {
    float l2_normal = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    for (int i = 0; i < 3; ++i) {
        v1[i] /= l2_normal;
    }
}

template <class T>
inline void PatchMatchAlg::cpy_vec3(T* dst, T* src) {
    memcpy(dst, src, 3 * sizeof(T));
}
