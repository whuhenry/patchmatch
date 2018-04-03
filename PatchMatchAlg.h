//
// Created by yanzhao on 2018/4/2.
//

#ifndef PATCHMATCH_PATCHMATCHALG_H
#define PATCHMATCH_PATCHMATCHALG_H

#include <memory>
#include "image.h"

class PatchMatchAlg {
public:
    float gamma_, alpha_, trunc_col_, trunc_grad_, max_dissimilarity;
    int window_radius_, neighbor_radius_, max_disparity_;
    PatchMatchAlg();
    void solve(std::shared_ptr<Image> imgL, std::shared_ptr<Image> imgR);
private:
    std::shared_ptr<Image> imgL_, imgR_;
    int rows_, cols_;
    int window_pixel_count_;
    std::default_random_engine random;
    enum MatchDirection{
        L2R = 1,
        R2L = -1
    };

    void random_init(std::shared_ptr<Image> img);
    void spatial_match(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                       int x, int y, bool ul_to_br, MatchDirection direction);
    void view_match(int x, int y, MatchDirection direction);
    void plane_refine(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                      int x, int y, MatchDirection direction);
    inline float disp_from_plane(int x, int y, cv::Vec3f& plane);
    float aggregated_cost(std::shared_ptr<Image> img1, std::shared_ptr<Image> img2,
                          int row, int col, cv::Vec3f &plane, MatchDirection direction);
    template<class T1, class T2>
    static float l1_norm(cv::Vec<T1, 3> v1, cv::Vec<T2, 3> v2);
    void show_result();
};


#endif //PATCHMATCH_PATCHMATCHALG_H
