#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>

#include "pcg32.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
// #define CHANNEL 3

// some const
inline constexpr __device__ float DENSITY_THRESH() { return 10.0f; } 
inline constexpr __device__ float SQRT3() { return 1.73205080757f; }

inline constexpr __device__ float MIN_NEAR() { return 0.05f; }
inline constexpr __device__ float SCALE() {return 1.0f;}
inline constexpr __device__ float DT_GAMMA() { return 1.f / 128.f; }
inline constexpr __device__ int MAX_STEPS() { return 2048; }
inline constexpr __device__ float MIN_STEPSIZE() { return (2 * SQRT3() / MAX_STEPS())*SCALE(); } 
#define CHANNEL_NUM 3
#define CHANNEL_PER 3
#define NTHREAD 256

// util functions
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(float x) {
	return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

__host__ __device__ void swapf(float& a, float& b) {
	float c = a; 
    a = b; 
    b = c;
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const scalar_t * __restrict__ grid,
    const float mean_density,
    const int iter_density,
    const float bound,
    const uint32_t N, const uint32_t H, const uint32_t M,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * xyzstorays,
    int * rays,
    int * counter,
    const bool perturb
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    const float rbound = 1 / bound;

    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching (naive, no mip, just one level)
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near_x = (-bound - ox) * rdx;
    float far_x = (bound - ox) * rdx;
    if (near_x > far_x) swapf(near_x, far_x);
    float near_y = (-bound - oy) * rdy;
    float far_y = (bound - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);
    float near_z = (-bound - oz) * rdz;
    float far_z = (bound - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    const float near = fmaxf(fmaxf(near_x, fmaxf(near_y, near_z)), 0.05f); 
    const float far = fminf(far_x, fminf(far_y, far_z));

    const float dt_min = MIN_STEPSIZE() * bound;
    const float dt_max = 2 * bound / (H - 1)*SCALE();
    const float dt_gamma = DT_GAMMA();

    const float t0 = near;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    pcg32 rng((uint64_t)n);

    while (t < far && num_steps < MAX_STEPS()) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);
        // convert to nearest grid position
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)+0.5); 
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1)+0.5);
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1)+0.5);

        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            num_steps++;
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            if (perturb) t += dt * rng.next_float() * 2;
            else t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }
    }
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    // write rays
    rays[n * 3] = n;
    rays[n * 3 + 1] = point_index;
    rays[n * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps >= M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index;
    xyzstorays += point_index;

    t = t0;
    uint32_t step = 0;

    rng = pcg32((uint64_t)n); // reset 

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        // convert to nearest grid position
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)+0.5);
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1)+0.5);
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1)+0.5);

        // query grid
        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;

            xyzstorays[0]=n;
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            if (perturb) {
                const float p_dt = dt * rng.next_float() * 2;
                t += p_dt;
                deltas[0] = p_dt;
            } else {
                t += dt;
                deltas[0] = dt;
            }
            xyzs += 3;
            dirs += 3;
            deltas++;
            step++;
            xyzstorays++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }

    }
}


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M]
// rays: [N, 3], idx, offset, num_steps
// depth: [N]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const scalar_t * __restrict__ bg_color,
    const int * __restrict__ rays,
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * depth,
    scalar_t * image,
    scalar_t * A
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps >= M) {
        depth[index] = 0;

        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            image[index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER+d] = bg_color[index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER+d];
        }
        A[index]=0;
        return;
    }



    sigmas += offset;
    rgbs += offset * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    deltas += offset;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;
    scalar_t sum_delta = 0; // sum of delta, to calculate the relative depth map.

    while (step < num_steps) {

        // minimal remained transmittence
        if (T < 1e-4f) 
        {break;}

        const scalar_t alpha = 1.0f - __expf(- (sigmas[0]) * deltas[0]);
        const scalar_t weight = alpha * T;

        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            image[index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER+d] += weight*rgbs[d];
        }

        T *= 1.0f - alpha;

        // locate
        sigmas++;
        rgbs += CHANNEL_NUM;
        deltas++;

        step++;
    }

    // mix with background
    if (step == num_steps) {
        // printf("");
        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            image[index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER+d] += T*bg_color[index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER+d];
        }
    
    }
    else{
        // printf("no bg\n");
    }
    A[index]=1-T;
}


// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M]
// rays: [N, 3], idx, offset, num_steps
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_img,
    const scalar_t * __restrict__ grad_A,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,  
    const int * __restrict__ rays,
    const scalar_t * __restrict__ image,  
    const scalar_t * __restrict__ A,
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps >= M) return;

    grad_img += index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    grad_A +=index;
    image += index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    A+=index;
    sigmas += offset;
    rgbs += offset * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    deltas += offset;
    grad_sigmas += offset;
    grad_rgbs += offset * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;

    scalar_t color_now[CHANNEL_PER];
    #pragma unroll
    for(uint32_t d=0;d<CHANNEL_PER;d++)
    {
        color_now[d]=0;
    }

    while (step < num_steps) {
        
        if (T < 1e-4f) break;

        const scalar_t alpha = 1.0f - __expf(- (sigmas[0]) * deltas[0]);
        const scalar_t weight = alpha * T;

        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            color_now[d]+=weight*rgbs[d];
        }

        T *= 1.0f - alpha; // this has been T(t+1)

        // write grad
        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            grad_rgbs[d]+=grad_img[d] * weight;
        }

        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            grad_sigmas[0]+=deltas[0] *grad_img[d] * (T * rgbs[d] - (image[d] - color_now[d]));
        }
        grad_sigmas[0]+=deltas[0] *(1-A[0])*grad_A[0];
    
        // locate
        sigmas++;
        rgbs += CHANNEL_NUM;
        grad_sigmas++;
        grad_rgbs += CHANNEL_NUM;
        deltas++;

        step++;
    }
}


void march_rays_train(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas,at::Tensor xyzstorays, at::Tensor rays, at::Tensor counter, const bool perturb) {
    CHECK_CUDA(rays_o);
    CHECK_CUDA(rays_d);
    CHECK_CUDA(grid);

    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    CHECK_CONTIGUOUS(grid);

    CHECK_IS_FLOATING(rays_o);
    CHECK_IS_FLOATING(rays_d);
    CHECK_IS_FLOATING(grid);

    static constexpr uint32_t N_THREAD = 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), mean_density, iter_density, bound, N, H, M, xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(),xyzstorays.data_ptr<int>(), rays.data_ptr<int>(), counter.data_ptr<int>(), perturb);
    }));
}


void composite_rays_train_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, const float bound, at::Tensor bg_color, const uint32_t M, const uint32_t N, at::Tensor depth, at::Tensor image,at::Tensor A) {

    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(deltas);
    CHECK_CUDA(rays);
    CHECK_CUDA(depth);
    CHECK_CUDA(image);
    CHECK_CUDA(bg_color);

    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(deltas);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(depth);
    CHECK_CONTIGUOUS(image);
    CHECK_CONTIGUOUS(bg_color);

    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(deltas);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(depth);
    CHECK_IS_FLOATING(image);
    CHECK_IS_FLOATING(bg_color);

    static constexpr uint32_t N_THREAD = NTHREAD;
    dim3 block(N_THREAD,div_round_up(CHANNEL_NUM,CHANNEL_PER));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), block>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), bg_color.data_ptr<scalar_t>(), rays.data_ptr<int>(), bound, M, N, depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(),A.data_ptr<scalar_t>());
    }));
}


void composite_rays_train_backward(at::Tensor grad_img,
at::Tensor grad_A, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, at::Tensor image,at::Tensor A, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {

    CHECK_CUDA(grad_img);
    CHECK_CUDA(grad_A);
    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(deltas);
    CHECK_CUDA(rays);
    CHECK_CUDA(image);
    CHECK_CUDA(grad_sigmas);
    CHECK_CUDA(grad_rgbs);

    CHECK_CONTIGUOUS(grad_img);
    CHECK_CONTIGUOUS(grad_A);
    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(deltas);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(image);
    CHECK_CONTIGUOUS(grad_sigmas);
    CHECK_CONTIGUOUS(grad_rgbs);

    CHECK_IS_FLOATING(grad_img);
    CHECK_IS_FLOATING(grad_A);
    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(deltas);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(image);
    CHECK_IS_FLOATING(grad_sigmas);
    CHECK_IS_FLOATING(grad_rgbs);

    static constexpr uint32_t N_THREAD = NTHREAD;
    dim3 block(N_THREAD,div_round_up(CHANNEL_NUM,CHANNEL_PER));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_img.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), block>>>(grad_img.data_ptr<scalar_t>(),grad_A.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), image.data_ptr<scalar_t>(),A.data_ptr<scalar_t>(), bound, M, N, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const float bound,
    const uint32_t H,
    const scalar_t * __restrict__ grid,
    const float mean_density,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    float t = rays_t[n]; // current ray's t

    const float rbound = 1 / bound;
    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);
    // const float density_thresh = fmaxf(DENSITY_THRESH(), mean_density);

    int flag=0;

    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = nears[index], far = fars[index];

    const float dt_min = MIN_STEPSIZE() * bound;
    const float dt_max = (2 * bound / (H - 1))*SCALE();
    const float dt_gamma = DT_GAMMA();

    // march for n_step steps, record points
    uint32_t step = 0;
    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        // convert to nearest grid position
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)+0.5); 
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1)+0.5);
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1)+0.5);

        // printf("(1,");

        // query grid
        const uint32_t g_index = nx * H * H + ny * H + nz;
        const float density = grid[g_index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            t += dt;
            deltas[0] = dt; // used to calc alpha
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;
        } else {
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }
    }
}

void march_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor rays_o, at::Tensor rays_d, const float bound, const uint32_t H, at::Tensor density_grid, const float mean_density, at::Tensor near, at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas) {
    static constexpr uint32_t N_THREAD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), bound, H, density_grid.data_ptr<scalar_t>(), mean_density, near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    scalar_t t = rays_t[n]; // current ray's t

    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    deltas += n * n_step * 2;

    weights_sum += index;
    depth += index;
    image += index * CHANNEL_NUM+threadIdx.y*CHANNEL_PER;
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t color_now[CHANNEL_PER];
    #pragma unroll
    for(uint32_t d=0;d<CHANNEL_PER;d++)
    {
        color_now[d]=image[d];
    }

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        if (deltas[0] == 0) break;
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        #pragma unroll
        for(uint32_t d=0;d<CHANNEL_PER;d++)
        {
            color_now[d]+=weight*rgbs[d];
        }

        // ray is terminated if T is too small
        if (T < 1e-2) break;

        // locate
        sigmas++;
        rgbs += CHANNEL_NUM;
        deltas += 2;
        step++;
    }
    // rays_t = -1 means ray is terminated early.
    if (step < n_step) {
        rays_t[n] = -1;
    } else {
        rays_t[n] = t;
    }

    weights_sum[0] = weight_sum;
    depth[0] = d;
    #pragma unroll
    for(uint32_t d=0;d<CHANNEL_PER;d++)
    {
        image[d]=color_now[d];
    }
}

void composite_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = NTHREAD;
    dim3 block(N_THREAD,div_round_up(CHANNEL_NUM,CHANNEL_PER));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), block>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_compact_rays(
    const uint32_t n_alive, 
    int* rays_alive, 
    const int* __restrict__ rays_alive_old, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ rays_t_old, 
    int* alive_counter
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    // rays_t_old[n] < 0 means ray died in last composite kernel.
    if (rays_t_old[n] >= 0) {
        const int index = atomicAdd(alive_counter, 1);
        rays_alive[index] = rays_alive_old[n];
        rays_t[index] = rays_t_old[n];
    }
}


void compact_rays(const uint32_t n_alive, at::Tensor rays_alive, at::Tensor rays_alive_old, at::Tensor rays_t, at::Tensor rays_t_old, at::Tensor alive_counter) {
    static constexpr uint32_t N_THREAD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_t.scalar_type(), "compact_rays", ([&] {
        kernel_compact_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, rays_alive.data_ptr<int>(), rays_alive_old.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_t_old.data_ptr<scalar_t>(), alive_counter.data_ptr<int>());
    }));
}