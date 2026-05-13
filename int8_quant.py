"""INT8 tensorwise quantization ops for ComfyUI-compatible model loading.

Adapted from: https://github.com/BobJohnson24/ComfyUI-INT8-Fast (AGPL-3.0)
Credit: dxqb, BobJohnson24, silveroxides, newgrit1004
"""

from __future__ import annotations

import json
import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import comfy.model_patcher
import comfy.lora
import comfy.utils

log = logging.getLogger(__name__)

try:
    from .int8_fused_kernel import triton_int8_linear, triton_int8_linear_per_row, triton_quantize_rowwise
    _TRITON_AVAILABLE = True
except Exception as _e:
    _TRITON_AVAILABLE = False
    log.warning("INT8: Triton not available (%s), falling back to torch._int_mm", _e)

# Runtime toggle (mirrors Int8TensorwiseOps.use_triton)
_use_triton = True

# ConvRot group size — must be a power of 4
CONVROT_GROUP_SIZE = 256


# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale


# ---------------------------------------------------------------------------
# W8A8 forward (slow path fallback)
# ---------------------------------------------------------------------------

@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale, bias, compute_dtype: torch.dtype) -> Tensor:
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


@torch.no_grad()
def int8_forward_dynamic_per_row(x: Tensor, weight: Tensor, weight_scale: Tensor, bias, compute_dtype: torch.dtype) -> Tensor:
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


# ---------------------------------------------------------------------------
# Int8TensorwiseOps — comfy custom_operations replacement
# ---------------------------------------------------------------------------

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """Custom ComfyUI operations for INT8 tensorwise quantization.

        Usage:
          model_options = {"custom_operations": Int8TensorwiseOps}
          # for pre-quantized checkpoints:
          Int8TensorwiseOps.dynamic_quantize = False
          # for on-the-fly quantization of BF16 checkpoints:
          Int8TensorwiseOps.dynamic_quantize = True
          Int8TensorwiseOps.excluded_names = [...]  # sensitive layers to skip
        """

        excluded_names: list[str] = []
        dynamic_quantize: bool = False
        enable_convrot: bool = False
        use_triton: bool = True
        dynamic_lora: bool = False
        lora_patches: dict = {}
        lora_strength: float = 1.0

        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer('weight_scale', None)
                self._is_quantized = False
                self._is_per_row = False
                self._use_convrot = False
                self._weight_scale_scalar = None
                self.compute_dtype = torch.bfloat16
                self.lora_patches = []

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):

                def normalize_key(key):
                    if not isinstance(key, str):
                        return key
                    for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                        if key.startswith(p):
                            return key[len(p):]
                    return key

                def pop_metadata(sd, p, k):
                    v = sd.pop(p + k, None)
                    if v is not None:
                        return v
                    v = sd.pop("model." + p + k, None)
                    if v is not None:
                        return v
                    if p.startswith("model."):
                        v = sd.pop(p[6:] + k, None)
                        if v is not None:
                            return v
                    if p.startswith("diffusion_model."):
                        v = sd.pop("diffusion_model." + p + k, None)
                        if v is not None:
                            return v
                    return None

                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"

                weight_scale = pop_metadata(state_dict, prefix, "weight_scale")
                comfy_quant_tensor = pop_metadata(state_dict, prefix, "comfy_quant")
                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)
                _ = state_dict.pop(input_scale_key, None)  # pop but ignore

                if comfy_quant_tensor is not None:
                    try:
                        quant_conf = json.loads(bytes(comfy_quant_tensor.tolist()).decode('utf-8'))
                        if quant_conf.get("convrot", False):
                            self._use_convrot = True
                            Int8TensorwiseOps.enable_convrot = True
                            if "convrot_groupsize" in quant_conf:
                                self._convrot_groupsize = quant_conf["convrot_groupsize"]
                                Int8TensorwiseOps._global_convrot_groupsize = self._convrot_groupsize
                    except Exception:
                        pass

                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Pre-quantized INT8 checkpoint
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)

                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self._weight_scale_scalar = weight_scale.float().item()
                                self.weight_scale = None
                                self._is_per_row = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = True
                            else:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = False
                        else:
                            self._weight_scale_scalar = float(weight_scale)
                            self.weight_scale = None
                            self._is_per_row = False

                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn):
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1

                        if is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # On-the-fly quantization
                            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                            if not hasattr(Int8TensorwiseOps, '_logged_otf'):
                                log.info(
                                    "INT8: On-the-fly quantization (ConvRot=%s)",
                                    getattr(Int8TensorwiseOps, 'enable_convrot', False),
                                )
                                Int8TensorwiseOps._logged_otf = True

                            w_gpu = weight_tensor.to(dev, non_blocking=True).float()
                            self._use_convrot = False
                            if getattr(Int8TensorwiseOps, 'enable_convrot', False) and self.in_features % CONVROT_GROUP_SIZE == 0:
                                try:
                                    from .convrot import build_hadamard, rotate_weight
                                    H = build_hadamard(CONVROT_GROUP_SIZE, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=CONVROT_GROUP_SIZE)
                                    self._use_convrot = True
                                except Exception as e:
                                    log.warning("INT8: ConvRot error: %s", e)

                            q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)
                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.register_buffer('weight_scale', q_scale.cpu())
                            self._weight_scale_scalar = None
                            self._is_quantized = True
                            self._is_per_row = True
                    else:
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

                # Update comfy's archived dtype so VBAR geometry uses correct sizes
                if self.weight is not None:
                    self.weight_comfy_model_dtype = self.weight.dtype
                if self.weight_scale is not None:
                    self.weight_scale_comfy_model_dtype = self.weight_scale.dtype
                if self.bias is not None:
                    self.bias_comfy_model_dtype = self.bias.dtype

            def _get_weight_scale(self):
                if self._weight_scale_scalar is not None:
                    return self._weight_scale_scalar
                return self.weight_scale

            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                # Dequantize for LoRA delta calculation in patch_weight_to_device.
                # cast_to_device already cast us to float; we still dequantize from
                # self.weight so the scale is correctly applied.
                w_scale = self._get_weight_scale()
                return dequantize(self.weight.float(), w_scale)

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight
                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                new_weight = quantize_int8(out_weight, self._get_weight_scale())
                if return_weight:
                    return new_weight
                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                if return_weight:
                    return out_bias
                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                need_cast = (
                    self.comfy_cast_weights
                    or len(self.weight_function) > 0
                    or len(self.bias_function) > 0
                )

                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    else:
                        return F.linear(x, self.weight, self.bias)

                # INT8 quantized path — if LoRA patches are present (either via
                # weight_function for the lowvram ModelPatcher path, or via
                # weight_lowvram_function for the Dynamic VBAR path), addmm_cuda
                # can't operate on INT8 (Char) tensors. Dequantize, apply LoRA delta,
                # re-quantize back to INT8, then run the normal INT8 forward path.
                _lowvram_fn = getattr(self, 'weight_lowvram_function', None)
                if len(self.weight_function) > 0 or _lowvram_fn is not None:
                    w_scale_orig = self._get_weight_scale()
                    w = self.weight.to(dtype=torch.float32, device=x.device)
                    ws = w_scale_orig.to(x.device) if isinstance(w_scale_orig, torch.Tensor) else w_scale_orig
                    w_fp = dequantize(w, ws)
                    for f in self.weight_function:
                        w_fp = f(w_fp)
                    if _lowvram_fn is not None:
                        w_fp = _lowvram_fn(w_fp.to(x.device))
                    # Re-quantize along rows (same layout as pre-quantized checkpoint)
                    q_weight, q_scale = quantize_int8_axiswise(w_fp.float(), dim=1)
                    b_fp = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
                    _bias_lowvram_fn = getattr(self, 'bias_lowvram_function', None)
                    for f in self.bias_function:
                        b_fp = f(b_fp)
                    if _bias_lowvram_fn is not None and b_fp is not None:
                        b_fp = _bias_lowvram_fn(b_fp.to(x.device))
                    compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                    x_shape = x.shape
                    x_2d = x.reshape(-1, x_shape[-1])
                    y = int8_forward_dynamic_per_row(x_2d, q_weight, q_scale, b_fp, compute_dtype)
                    # Additive LoRA: registered at load time, bypasses re-quant overhead
                    for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                        lD = lora_down.to(x.device, non_blocking=True)
                        lU = lora_up.to(x.device, non_blocking=True)
                        lora_x = F.linear(x_2d.to(lD.dtype), lD)
                        lora_y = F.linear(lora_x, lU)
                        if lora_start is not None:
                            y[:, lora_start:lora_start + lora_size] = (
                                y[:, lora_start:lora_start + lora_size] + lora_y.to(y.dtype)
                            )
                        else:
                            y = y + lora_y.to(y.dtype)
                    return y.reshape(*x_shape[:-1], y.shape[-1])

                if need_cast:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True,
                    )
                else:
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None

                w_scale = self._get_weight_scale()
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])

                if getattr(self, "_use_convrot", False):
                    try:
                        from .convrot import build_hadamard, rotate_activation
                        group_size = getattr(self, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                        H = build_hadamard(group_size, device=x.device, dtype=x.dtype)
                        x_2d = rotate_activation(x_2d, H, group_size=group_size)
                    except Exception:
                        pass

                # Sync triton toggle
                import sys as _sys
                _mod = _sys.modules[__name__]
                _mod._use_triton = Int8TensorwiseOps.use_triton

                if x_2d.shape[0] > 16:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    # Small batch fallback (dequant)
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)

                # Dynamic LoRA
                for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                    lD = lora_down.to(x.device, non_blocking=True)
                    lU = lora_up.to(x.device, non_blocking=True)
                    lora_x = F.linear(x_2d.to(lD.dtype), lD)
                    lora_y = F.linear(lora_x, lU)
                    if lora_start is not None:
                        y[:, lora_start:lora_start + lora_size] = (
                            y[:, lora_start:lora_start + lora_size] + lora_y.to(y.dtype)
                        )
                    else:
                        y = y + lora_y.to(y.dtype)

                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)
                return y.reshape(*x_shape[:-1], y.shape[-1])

        # Pass-through for non-Linear layers
        class GroupNorm(manual_cast.GroupNorm): pass
        class LayerNorm(manual_cast.LayerNorm): pass
        class Conv2d(manual_cast.Conv2d): pass
        class Conv3d(manual_cast.Conv3d): pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding): pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")

else:
    Int8TensorwiseOps = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LTX2-specific excluded layer names (sensitive to quantization)
# ---------------------------------------------------------------------------
LTX2_EXCLUDED_NAMES = [
    'adaln_single', 'audio_adaln_single', 'audio_caption_projection',
    'audio_patchify_proj', 'audio_proj_out', 'audio_scale_shift_table',
    'av_ca_a2v_gate_adaln_single', 'av_ca_audio_scale_shift_adaln_single',
    'av_ca_v2a_gate_adaln_single', 'av_ca_video_scale_shift_adaln_single',
    'caption_projection', 'patchify_proj', 'proj_out', 'scale_shift_table',
]
