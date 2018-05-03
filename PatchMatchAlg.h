//
// Created by yanzhao on 2018/4/2.
//

#ifndef PATCHMATCH_PATCHMATCHALG_H
#define PATCHMATCH_PATCHMATCHALG_H

#include <memory>
#include <random>
#include "Image.h"

class PatchMatchAlg {
public:
    float gamma_, alpha_, trunc_col_, trunc_grad_;
    int window_radius_, max_disparity_;
    PatchMatchAlg();
    ~PatchMatchAlg();
    void solve(std::shared_ptr<Image> imgL, std::shared_ptr<Image> imgR);
    void save_disp_map(std::string path);
    void write_result(std::string dir);
private:
    Image *imgL_, *imgR_;
    int rows_, cols_;
    int window_pixel_count_;
    float max_dissimilarity;
    cv::Mat disp_mat_, disp_mat_unfilter_, mask_mat_;
    std::default_random_engine random;
    enum MatchDirection{
        L2R = 1,
        R2L = -1
    };

    void random_init(Image *img);
    void spatial_match(int iter_num);
    void view_match(int iter_num);
    void plane_refine(int iter_num);
    void post_process();
    inline float disp_from_plane(int x, int y, float *plane);
    float aggregated_cost(Image* img1, Image* img2, int y, int x, float *plane,
                          MatchDirection direction, float base_cost = FLT_MAX);

    template <class T1, class T2>
    static float l1_distance(T1 *v1, T2 *v2);

    template <class T>
    static void l2_norm(T * v1);

    template <class T>
    static void cpy_vec3(T *dst, T *src);
    void show_result();
};


#endif //PATCHMATCH_PATCHMATCHALG_H
