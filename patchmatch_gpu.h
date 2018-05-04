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

struct cuImage {
    float* d_image;
    float* d_plane;
    float* d_normal;
    float* d_grad;
    float* d_cost;
};

void cpy_host_image_to_cuimage(Image* host_im, cuImage* cu_im);
__global__ void initNormalAndPlane(float* normal, float* plane, float* cost, curandState* globalState,
                                   PatchMatchConfig cfg, int pixel_per_thread);
__global__ void spatialPropagation(cuImage im_base, cuImage im_ref,
                                   PatchMatchConfig cfg, int red_or_black, int direction);

__global__ void setup_kernel(curandState* state, unsigned long long seed);

__global__ void init_cost(cuImage im_base, cuImage im_ref, PatchMatchConfig cfg, int direction);

__device__ float compute_cost_cu(cuImage* im_base, cuImage* im_ref, int x, int y,
                                 float* plane_used, int direction, PatchMatchConfig *cfg);

__device__ void norm(float* v);

__device__ float l1_distance(float *v1, float *v2);

__device__ void normal_to_plane(int x, int y, float z, float* normal, float *plane);

__device__ float plane_to_disp(int x, int y, float* plane_single);

__device__ void cpy_vec3(float* dst, float* src);

__device__ void get_value_bilinear(float x, float y, PatchMatchConfig* cfg, float* in, float* out);


#endif
