#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings,at::Tensor xyzstoframes, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, const uint32_t gridtype);
void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings,at::Tensor xyzstoframes, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const uint32_t gridtype);

#endif