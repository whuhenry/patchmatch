#include "patchmatch_gpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void init(Image* im, PatchMatchConfig cfg) {
    int pixel_count = im->rows_ * im->cols_;
    cudaMalloc(&(im->d_image_), pixel_count * 3 * sizeof(float));
    cudaMemcpy(im->d_image_, im->image_, pixel_count * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&(im->d_plane_), pixel_count * 3 * sizeof(float));
    cudaMalloc(&(im->d_normal_), pixel_count * 3 * sizeof(float));
    cudaMalloc(&(im->d_grad_), pixel_count * 3 * sizeof(float));
    cudaMalloc(&(im->d_cost_), pixel_count * sizeof(float));

    dim3 grid_size, blockdim(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
    grid_size.x = (cfg.cols + blockdim.x - 1) / blockdim.x;
    grid_size.y = (cfg.cols + blockdim.y - 1) / blockdim.y;
    curandState *devStates;
    cudaMalloc((void **)&devStates, blockdim.x * blockdim.y * sizeof(curandState));

    initNormalAndPlane<<<grid_size, blockdim>>>(im->d_normal_, im->d_plane_, devStates, cfg);
    
}

__global__ void initNormalAndPlane(float* normal, float* plane, curandState* globalState, PatchMatchConfig cfg) {
    int x      = blockIdx.x * blockDim.x + threadIdx.x;
    int y      = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * cfg.cols + x;

    curandState local_state;
    local_state = globalState[threadIdx.x];
    
    float disp = curand_uniform(&local_state) * cfg.max_disp;
    normal[offset]     = (curand_uniform(&local_state) - 0.5f) * 2;
    normal[offset + 1] = (curand_uniform(&local_state) - 0.5f) * 2;
    normal[offset + 2] = curand_uniform(&local_state);
    norm(normal + offset);

    normal_to_plane(x, y, disp, normal + offset, plane + offset);
}