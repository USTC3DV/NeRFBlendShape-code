#pragma once

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>
// inference functions
void march_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor rays_o, at::Tensor rays_d, const float bound, const uint32_t H, at::Tensor density_grid, const float mean_density, at::Tensor near, at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image);
void compact_rays(const uint32_t n_alive, at::Tensor rays_alive, at::Tensor rays_alive_old, at::Tensor rays_t, at::Tensor rays_t_old, at::Tensor alive_counter);