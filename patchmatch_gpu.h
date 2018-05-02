//
// Created by yanzhao on 2018/5/1.
//

#ifndef PATCHMATCH_PATCHMATCH_GPU_H
#define PATCHMATCH_PATCHMATCH_GPU_H

#include <curand.h>
#include <curand_kernel.h>
#include "Image.h"

struct PatchMatchConfig {
    int rows;
    int cols;
    int max_disp;
    float alpha;
    float gamma;
    float density_diff_max;
    float grad_diff_max;
    int window_radius;
    int2 *h_neighbor_lists;
    int2 *d_neighbor_lists;
    int neighbor_lists_len;
    float max_cost_single;
    int iter_count;
};

const int BLOCK_DIM_SIZE = 32;

void init(Image* im, PatchMatchConfig cfg);
void solve(Image *im_left, Image *im_right, PatchMatchConfig cfg);

__global__ void initNormalAndPlane(float* normal, float* plane, float* cost, curandState* globalState,
                                   PatchMatchConfig cfg, int pixel_per_thread);
__global__ void spatialPropagation(Image *im_base, Image *im_ref, 
                                   PatchMatchConfig cfg, int red_or_black, int direction);

__global__ void setup_kernel(curandState* state);

__device__ float compute_cost_cu(Image* im_base, Image* im_ref, int x, int y, 
                                 float* plane_used, int direction, PatchMatchConfig *cfg);

__device__ inline void norm(float* v) {
    float l2_normal = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l2_normal;
    v[1] /= l2_normal;
    v[2] /= l2_normal;
}

__device__ inline float l1_distance(float *v1, float *v2) {
    return abs(v1[0] - v2[0]) + abs(v1[1] - v2[1]) + abs(v1[2] - v2[2]);
}

__device__ inline void normal_to_plane(int x, int y, float z, float* normal, float *plane) {
    plane[0] = -normal[0] / normal[2];
    plane[1] = -normal[1] / normal[2];
    plane[2] = (normal[0] * x + normal[1] * y + normal[2] * z) / normal[2];
}

__device__ inline float plane_to_disp(int x, int y, float* plane_single) {
    return x * plane_single[0] + y * plane_single[1] + plane_single[2];
}

__device__ inline void cpy_vec3(float* dst, float* src) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

__device__ inline void get_value_bilinear(float x, float y, PatchMatchConfig* cfg, float* in, float* out) {
    auto ix = int(x);
    auto iy = int(y);
    auto center = (iy * cfg->cols + ix) * 3;
    float u = y - iy;
    float v = x - ix;
    if (iy == y && ix == x) {
        for (int i = 0; i < 3; i++) {
            out[i] = in[center + i];
        }
    }
    else if (iy == y && ix != x) {
        for (int i = 0; i < 3; i++) {
            out[i] = (1 - v) * in[center + i] + v * in[center + i + 3];
        }
    }
    else if (iy != y && ix == x) {
        for (int i = 0; i < 3; i++) {
            out[i] = (1 - u) * in[center + i] + u * in[center + i + 3 * cfg->cols];
        }
    }
    else {
        for (int i = 0; i < 3; i++) {
            out[i] = (1 - u) * (1 - v) * in[center + i] + (1 - u) * v * in[center + i + 3]
                + u * (1 - v) * in[center + i + 3 * cfg->cols] + u * v * in[center + i + 3 * cfg->cols + 3];
        }
    }
}


#endif
