//
// Created by yanzhao on 2018/5/1.
//

#ifndef PATCHMATCH_PATCHMATCH_GPU_H
#define PATCHMATCH_PATCHMATCH_GPU_H

#include <curand_kernel.h>
#include "Image.h"

struct PatchMatchConfig {
    int rows;
    int cols;
    int max_disp;
    int alpha;
    int gamma;
    int density_diff_max;
    int grad_diff_max;
};

const int BLOCK_DIM_SIZE = 32;

void init(Image* im, PatchMatchConfig cfg);

__global__ void initNormalAndPlane(float* normal, float* plane, curandState* globalState, PatchMatchConfig cfg);

__device__ inline void norm(float* v) {
    float l2_normal = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; ++i) {
        v[i] /= l2_normal;
    }
}

__device__ inline void normal_to_plane(int x, int y, float z, float* normal, float *plane) {
    plane[0] = -normal[0] / normal[2];
    plane[1] = -normal[1] / normal[2];
    plane[2] = (normal[0] * x + normal[1] * y + normal[2] * z) / normal[2];
}

#endif
