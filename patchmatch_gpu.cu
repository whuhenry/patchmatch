#include "patchmatch_gpu.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

void init(Image* im, PatchMatchConfig cfg) {
    int pixel_count = im->rows_ * im->cols_;
    checkCudaErrors(cudaMalloc(&(im->d_image_), pixel_count * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(im->d_image_, im->image_, pixel_count * 3 * sizeof(float), cudaMemcpyHostToDevice));

    im->plane_ = new float[pixel_count * 3];
    for (int i = 0; i < pixel_count * 3; ++i) {
        im->plane_[i] = 0.0f;
    }

    checkCudaErrors(cudaMalloc(&(im->d_plane_), pixel_count * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(im->d_plane_, im->plane_, pixel_count * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&(im->d_normal_), pixel_count * 3 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&(im->d_grad_), pixel_count * 3 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&(im->d_cost_), pixel_count * sizeof(float)));

    
    curandState *devStates;
    cudaError_t err = cudaMalloc(&devStates, 64 * 64 * sizeof(curandState));
    setup_kernel<<<64, 64>>>(devStates);
 
    int pixel_count_per_thread = (cfg.rows * cfg.cols + 64 * 64 - 1) / (64 * 64);
    initNormalAndPlane<<<64, 64>>>(im->d_normal_, im->d_plane_, im->d_cost_, devStates, cfg, pixel_count_per_thread);
    checkCudaErrors(cudaFree(devStates));

//    checkCudaErrors(cudaMemcpy(im->plane_, im->d_plane_, pixel_count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
//    cv::Mat disp(cfg.rows, cfg.cols, CV_8U);
//    int offset = 0;
//    for (int i = 0; i < cfg.rows; ++i) {
//        for (int j = 0; j < cfg.cols; ++j) {
//            disp.at<uint8_t>(i, j) = (uint8_t)(j * im->plane_[offset] + i * im->plane_[offset + 1] + im->plane_[offset + 2] / cfg.max_disp * 255.0f);
//            offset += 3;
//        }
//    }
//    cv::imshow("disp", disp);
//    cv::waitKey(0);
}

void solve(Image * im_left, Image * im_right, PatchMatchConfig cfg)
{
    dim3 grid_size, blockdim(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
    grid_size.x = (cfg.cols + blockdim.x - 1) / blockdim.x;
    grid_size.y = (cfg.rows + blockdim.y - 1) / blockdim.y;
    cuImage cuim_left, cuim_right;
    cpy_host_image_to_cuimage(im_left, &cuim_left);
    cpy_host_image_to_cuimage(im_right, &cuim_right);

    checkCudaErrors(cudaMalloc(&cfg.d_neighbor_lists, cfg.neighbor_lists_len * sizeof(int2)));
    checkCudaErrors(cudaMemcpy(cfg.d_neighbor_lists, cfg.h_neighbor_lists, 
                               cfg.neighbor_lists_len * sizeof(int2), cudaMemcpyHostToDevice));

    for (int iter = 0; iter < 1/*cfg.iter_count*/; ++iter) {
        //left to right red
        spatialPropagation<<<grid_size, blockdim>>>(cuim_left, cuim_right, cfg, 0, 1);
        ////left to right black
        spatialPropagation <<<grid_size, blockdim>>>(cuim_left, cuim_right, cfg, 1, 1);
        ////right to left black
        //spatialPropagation <<<grid_size, blockdim>>>(im_right, im_left, cfg, 0, -1);
        ////right to left black
        //spatialPropagation <<<grid_size, blockdim>>>(im_right, im_left, cfg, 1, -1);
    }
    //cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(im_left->plane_, im_left->d_plane_, 
                               cfg.rows * cfg.cols * 3 * sizeof(float), cudaMemcpyDeviceToHost));
//    cv::Mat disp(cfg.rows, cfg.cols, CV_8U);
//    int offset = 0;
//    for (int i = 0; i < cfg.rows; ++i) {
//        for (int j = 0; j < cfg.cols; ++j) {
//            disp.at<uint8_t>(i, j) = (uint8_t)(j * im_left->plane_[offset] + i * im_left->plane_[offset + 1]
//                                               + im_left->plane_[offset + 2] / cfg.max_disp * 255.0f);
//            offset += 3;
//        }
//    }
//    cv::imshow("disp", disp);
//    cv::waitKey(0);
}

void cpy_host_image_to_cuimage(Image* host_im, cuImage* cu_im) {
    cu_im->d_image = host_im->d_image_;
    cu_im->d_plane = host_im->d_plane_;
    cu_im->d_normal = host_im->d_normal_;
    cu_im->d_grad = host_im->d_grad_;
    cu_im->d_cost = host_im->d_cost_;
}

__global__ void setup_kernel(curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(0, id, 0, &state[id]);
}

__global__ void initNormalAndPlane(float* normal, float* plane, float* cost, curandState* globalState,
                                   PatchMatchConfig cfg, int pixel_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_offset = tid * pixel_per_thread;
    int offset = pixel_offset * 3;

    curandState local_state = globalState[tid];
    int y = pixel_offset / cfg.cols;
    int x = pixel_offset % cfg.cols;
    if (y >= cfg.rows) {
        return;
    }

    for (int i = 0; i < pixel_per_thread; ++i) {        
        float disp = curand_uniform(&local_state) * cfg.max_disp;
        normal[offset] = (curand_uniform(&local_state) - 0.5f) * 2;
        normal[offset + 1] = (curand_uniform(&local_state) - 0.5f) * 2;
        normal[offset + 2] = curand_uniform(&local_state);
        norm(normal + offset);

        normal_to_plane(x, y, disp, normal + offset, plane + offset);
        cost[pixel_offset] = cfg.max_cost_single * (cfg.window_radius * 2 + 1) * (cfg.window_radius * 2 + 1);
        offset += 3;
        ++x;
        if (x >= cfg.cols) {
            x = 0;
            ++y;
            if(y >= cfg.rows) {
                return;
            }
        }
    }
}

__global__ void spatialPropagation(cuImage im_base, cuImage im_ref, PatchMatchConfig cfg, int red_or_black, int direction) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cfg.cols || y >= cfg.rows) {
        return;
    }
    int offset = y * cfg.cols + x;

    //chess board update: detail see paper "Massively Parallel Multiview Stereopsis by Surface Normal Diffusion"
    if((y % 2 + x % 2) % 2 != red_or_black) {
        return;
    }

    float* center_plane = &im_base.d_plane[offset * 3];
    float* max_cost = &im_base.d_cost[offset];
    for (int i = 0; i < cfg.neighbor_lists_len; ++i) {
        int ny = y + cfg.d_neighbor_lists[i].y;
        int nx = x + cfg.d_neighbor_lists[i].x;
        if (ny <= 0 || ny >= cfg.rows || nx <= 0 || ny >= cfg.cols) {
            continue;
        }
        float* comp_plane = &im_base.d_plane[(ny * cfg.cols + nx) * 3];
        float cost = compute_cost_cu(&im_base, &im_ref, x, y, comp_plane, direction, &cfg);
        
        if (cost < max_cost[0]) {
            max_cost[0] = cost;
            cpy_vec3(center_plane, comp_plane);
        }
    }

//    if (x > 400 && y >300) {
//        float weight = im_base.d_plane[(y * cfg.cols + x) * 3] - im_base.d_plane[(y * cfg.cols + x) * 3];
//        //printf("%d, %d, %f\n", x, y, im_base.d_plane[(y * cfg.cols + x) * 3] - im_base.d_plane[(y * cfg.cols + x) * 3]);
//        im_base.d_plane[offset * 3] = 0.0f;
//        im_base.d_plane[offset * 3 + 1] = 0.0f;
//        im_base.d_plane[offset * 3 + 2] = 0.0f;
//    }
    
}

__device__ float compute_cost_cu(cuImage* im_base, cuImage* im_ref, int x, int y,
                                 float* plane_used, int direction, PatchMatchConfig *cfg) {
    if (y - cfg->window_radius < 0 || y + cfg->window_radius > cfg->rows - 1
        || x - cfg->window_radius < 0 || x + cfg->window_radius > cfg->cols - 1) {
        return cfg->max_cost_single * (cfg->window_radius * 2 + 1) * (cfg->window_radius * 2 + 1);
    } else {
        int x_st = x - cfg->window_radius;
        int x_ed = x + cfg->window_radius;
        int y_st = y - cfg->window_radius;
        int y_ed = y + cfg->window_radius;
        float sum_cost = 0.0f;
        int center_offset = (y * cfg->cols + x) * 3;
        float density_ref_single[3], grad_ref_single[3];
        int cw_offset;
        float weight, disp, cor_x;
        for (int cw_y = y_st; cw_y <= y_ed; ++cw_y) {
            for (int cw_x = x_st; cw_x <= x_ed; ++cw_x) {
                cw_offset = (cw_y * cfg->cols + cw_x) * 3;
                //weight = im_base->d_plane[center_offset] - im_base->d_plane[center_offset];
                //float weight = abs(im_base.d_plane[center_offset] - im_base.d_plane[cw_offset]);
                weight = exp(-l1_distance(&im_base->d_image[center_offset], &im_base->d_image[cw_offset]) / cfg->gamma);
                disp = plane_to_disp(cw_x, cw_y, plane_used);
                cor_x = cw_x - disp * direction;
                if (cor_x < 0 || cor_x > cfg->cols - 1) {
                    sum_cost += weight * cfg->max_cost_single;
                } else {
                    get_value_bilinear(cor_x, cw_y, cfg, im_ref->d_image, density_ref_single);
                    get_value_bilinear(cor_x, cw_y, cfg, im_ref->d_grad, grad_ref_single);
                    sum_cost += weight * (
                        (1 - cfg->alpha)
                        * MIN(l1_distance(im_base->d_image + cw_offset, density_ref_single), cfg->density_diff_max)
                        + cfg->alpha
                        * MIN(l1_distance(im_base->d_grad + cw_offset, grad_ref_single), cfg->grad_diff_max));
                }
            }
        }
        return sum_cost;
    }
}
