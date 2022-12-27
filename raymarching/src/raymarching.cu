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
inline constexpr __device__ float DENSITY_THRESH() { return 10.0f; } // TODO: how to decide this threshold ?//10.0f
inline constexpr __device__ float SQRT3() { return 1.73205080757f; }

inline constexpr __device__ float MIN_NEAR() { return 0.05f; }
inline constexpr __device__ float SCALE() {return 1.0f;}
inline constexpr __device__ float DT_GAMMA() { return 1.f / 128.f; }
inline constexpr __device__ int MAX_STEPS() { return 2048; }
inline constexpr __device__ float MIN_STEPSIZE() { return (2 * SQRT3() / MAX_STEPS())*SCALE(); } // still need to mul bound to get dt_min
#define CHANNEL_NUM 3
#define CHANNEL_PER 3
#define NTHREAD 512

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
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)+0.5); // (x + bound) / (2 * bound) * (H - 1); range in [0, H-1]
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1)+0.5);
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1)+0.5);

        // query grid
        const uint32_t g_index = nx * H * H + ny * H + nz;
        // printf("(%f,%f,%f,%d,%d,%d,%d)",x,y,z,nx,ny,nz,g_index);
        const float density = grid[g_index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            // write step
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

        // else, skip a large step (basically skip a voxel grid)
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
    static constexpr uint32_t N_THREAD = NTHREAD;
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
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
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