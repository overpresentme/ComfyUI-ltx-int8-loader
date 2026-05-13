"""Triton fused INT8 kernels.

Source: https://github.com/BobJohnson24/ComfyUI-INT8-Fast (AGPL-3.0)
Credit: dxqb (https://github.com/Nerogar/OneTrainer/pull/1034)
        BobJohnson24, newgrit1004, silveroxides
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# =============================================================================
# Kernel 1: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)
# =============================================================================

@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * n_elements
    y_row_ptr = y_ptr + row_idx * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    scale = tl.maximum(max_val / 127.0, 1e-30)
    q_f = x / scale
    q_i = libdevice.rint(q_f).to(tl.int32)
    q_i = tl.clamp(q_i, -128.0, 127.0)
    tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
    tl.store(s_ptr + row_idx, scale.to(tl.float32))


def triton_quantize_rowwise(x: torch.Tensor):
    """Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch, 1] (float32)"""
    rows, cols = x.shape
    y = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128
    grid = (rows,)
    _quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
    return y, s


# =============================================================================
# Kernel 2: INT8 GEMM + Fused Dequantization Epilogue (scalar weight scale)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_dequant_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)
    scale_b = tl.load(b_scale_ptr)
    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=torch.float16):
    """Fused pipeline for W8A8 Linear Layer (scalar weight scale)."""
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    x_int8, x_scale = triton_quantize_rowwise(x_2d)
    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    elif weight_scale.numel() == 1:
        weight_scale = weight_scale.reshape(1)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    _int8_matmul_dequant_kernel[grid](
        a_ptr=x_int8, b_ptr=weight, c_ptr=output,
        a_scale_ptr=x_scale, b_scale_ptr=weight_scale, bias_ptr=bias_ptr,
        M=M, N=N, K=K,
        stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        HAS_BIAS=has_bias
    )
    return output.reshape(x_shape_orig[:-1] + (N,))


# =============================================================================
# Kernel 3: INT8 GEMM + Fused Dequant with Per-Row Weight Scales
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_dequant_per_row_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)
    scale_b = tl.load(b_scale_ptr + offs_bn)
    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b[None, :]
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_int8_linear_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None, compute_dtype=torch.float16):
    """Fused pipeline for W8A8 Linear Layer with per-row weight quantization."""
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    x_int8, x_scale = triton_quantize_rowwise(x_2d)
    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)
    ws = weight_scale.reshape(N).contiguous()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    _int8_matmul_dequant_per_row_kernel[grid](
        a_ptr=x_int8, b_ptr=weight, c_ptr=output,
        a_scale_ptr=x_scale, b_scale_ptr=ws, bias_ptr=bias_ptr,
        M=M, N=N, K=K,
        stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        HAS_BIAS=has_bias
    )
    return output.reshape(x_shape_orig[:-1] + (N,))
