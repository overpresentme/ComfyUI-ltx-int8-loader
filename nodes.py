"""LTX-Video INT8 nodes for ComfyUI.

Provides INT8-aware checkpoint loading and additive LoRA support for LTX2,
based on the ltx_standalone project's int8 quantization approach.

Nodes:
  LTXInt8CheckpointLoader  — load full LTX2 checkpoint with INT8 quantization
  LTXInt8AdditiveLoRA      — apply LoRA additively on Int8Linear modules
                             (avoids dequant-requant roundtrip each forward pass)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.weight_adapter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LTX2 excluded layer names (sensitive to quantization; kept in fp16/bf16)
# ---------------------------------------------------------------------------
LTX2_EXCLUDED_NAMES: list[str] = [
    "adaln_single", "audio_adaln_single", "audio_caption_projection",
    "audio_patchify_proj", "audio_proj_out", "audio_scale_shift_table",
    "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single",
    "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
]


# ---------------------------------------------------------------------------
# Helpers (mirror standalone loader.py logic)
# ---------------------------------------------------------------------------

def _build_model_options(mode: str) -> dict:
    """Build comfy model_options for the requested quantization mode.

    mode:
      ``int8_tensorwise``     — pre-quantized INT8 checkpoint (e.g. Winnougan)
      ``int8_tensorwise_otf`` — on-the-fly INT8 quantisation of a BF16/FP8 ckpt
    """
    from .int8_quant import Int8TensorwiseOps

    if Int8TensorwiseOps is None:
        log.warning("INT8: comfy ops unavailable; loading without quantization.")
        return {}

    # Reset class-level state so a second load in the same session is clean
    Int8TensorwiseOps.excluded_names = list(LTX2_EXCLUDED_NAMES)
    Int8TensorwiseOps.enable_convrot = False
    Int8TensorwiseOps.use_triton = True
    Int8TensorwiseOps.dynamic_lora = False
    Int8TensorwiseOps.lora_patches = {}
    if hasattr(Int8TensorwiseOps, "_logged_otf"):
        delattr(Int8TensorwiseOps, "_logged_otf")

    if mode == "int8_tensorwise_otf":
        Int8TensorwiseOps.dynamic_quantize = True
        Int8TensorwiseOps.enable_convrot = True
        log.info("INT8: on-the-fly quantization + ConvRot (ltx2 exclusion list).")
    else:  # int8_tensorwise
        Int8TensorwiseOps.dynamic_quantize = False
        log.info("INT8: pre-quantized checkpoint mode.")

    return {"custom_operations": Int8TensorwiseOps}


def _register_lora_additive(model_patcher) -> None:
    """Convert standard weight-function LoRA patches to additive ops on
    Int8Linear modules.

    For each patched weight that belongs to an Int8Linear layer, the
    (lora_down, lora_up) tensors are stored directly on the module so the
    forward pass adds the low-rank delta *after* the INT8 matmul — no
    re-quantisation, no precision loss.

    Complex LoRA types (LoHA, LoKR, DoRA, mid) remain in model_patcher.patches
    and fall back to the standard weight_function path.
    """
    from .int8_quant import Int8TensorwiseOps

    if Int8TensorwiseOps is None:
        log.warning("_register_lora_additive: Int8TensorwiseOps unavailable.")
        return

    if not model_patcher.patches:
        return

    total_keys = len(model_patcher.patches)
    log.info("_register_lora_additive: inspecting %d patch keys.", total_keys)

    registered = 0
    kept_fallback = 0
    skip_not_weight = 0
    skip_get_attr = 0
    skip_not_int8 = 0
    skip_complex = 0
    keys_to_remove: list[str] = []

    for key in list(model_patcher.patches.keys()):
        parts = key.rsplit(".", 1)
        if len(parts) < 2 or parts[1] != "weight":
            skip_not_weight += 1
            continue
        module_path = parts[0]

        try:
            module = comfy.utils.get_attr(model_patcher.model, module_path)
        except Exception:
            skip_get_attr += 1
            continue

        if not isinstance(module, Int8TensorwiseOps.Linear):
            skip_not_int8 += 1
            continue

        new_lora_patches: list = []
        all_simple = True

        for p in model_patcher.patches[key]:
            # p = (strength_patch, v, strength_model, offset, function)
            strength = float(p[0])
            v = p[1]

            if not isinstance(v, comfy.weight_adapter.LoRAAdapter):
                all_simple = False
                skip_complex += 1
                break

            mat1, mat2, raw_alpha, mid, dora_scale, reshape = v.weights
            if mid is not None or dora_scale is not None or reshape is not None:
                all_simple = False
                skip_complex += 1
                break

            alpha = float(raw_alpha / mat2.shape[0]) if raw_alpha is not None else 1.0
            scale = strength * alpha

            lora_down = mat2.flatten(start_dim=1).clone().cpu()
            lora_up = (mat1.flatten(start_dim=1).float() * scale).to(mat1.dtype).clone().cpu()
            new_lora_patches.append((lora_down, lora_up, None, None))

        if all_simple and new_lora_patches:
            module.lora_patches = new_lora_patches
            keys_to_remove.append(key)
            registered += 1
        else:
            kept_fallback += 1

    for key in keys_to_remove:
        model_patcher.patches.pop(key, None)
        model_patcher.backup.pop(key, None)

    if registered or kept_fallback:
        model_patcher.patches_uuid = uuid.uuid4()

    log.info(
        "_register_lora_additive: registered=%d  fallback=%d  "
        "skip_not_weight=%d  skip_get_attr=%d  skip_not_int8=%d  skip_complex=%d",
        registered, kept_fallback,
        skip_not_weight, skip_get_attr, skip_not_int8, skip_complex,
    )


# ---------------------------------------------------------------------------
# Node 1: LTXInt8CheckpointLoader
# ---------------------------------------------------------------------------

class LTXInt8CheckpointLoader:
    """Load a full LTX2 checkpoint (diffusion model + VAE) with INT8 quantization.

    Supports both pre-quantized INT8 checkpoints (e.g. Winnougan
    ltx-2.3-22b-distilled-int8tensormixed.safetensors) and on-the-fly
    quantization of standard BF16/FP8 checkpoints.

    Returns MODEL and VAE.  Load the text encoder (Gemma) separately with
    DualCLIPLoader / CLIPLoader and pass it into the sampler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "mode": (
                    ["int8_tensorwise", "int8_tensorwise_otf"],
                    {
                        "default": "int8_tensorwise",
                        "tooltip": (
                            "int8_tensorwise: load a pre-quantized INT8 checkpoint.  "
                            "int8_tensorwise_otf: quantize a BF16/FP8 checkpoint on-the-fly."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/ltx"
    DESCRIPTION = (
        "Load an LTX2 checkpoint with INT8 tensorwise quantization. "
        "Load the Gemma text encoder separately with DualCLIPLoader."
    )

    def load_checkpoint(self, ckpt_name: str, mode: str):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        model_options = _build_model_options(mode)

        log.info("LTXInt8CheckpointLoader: loading %s (mode=%s)", ckpt_name, mode)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)

        out = comfy.sd.load_state_dict_guess_config(
            sd,
            output_vae=True,
            output_clip=False,
            output_clipvision=False,
            output_model=True,
            metadata=metadata,
            model_options=model_options,
        )
        if out is None:
            raise RuntimeError(
                f"Could not detect model type for checkpoint: {ckpt_name}"
            )

        model, _clip, vae, _clipvision = out
        del sd

        log.info("LTXInt8CheckpointLoader: done.")
        return (model, vae)


# ---------------------------------------------------------------------------
# Node 2: LTXInt8AdditiveLoRA
# ---------------------------------------------------------------------------

class LTXInt8AdditiveLoRA:
    """Apply a LoRA to an INT8-quantized LTX2 model using the additive path.

    Standard LoRA application dequantizes the INT8 weight, adds the delta,
    then re-quantizes — losing precision on every forward pass.

    This node instead registers (lora_down, lora_up) directly on each
    Int8Linear module so the delta is added *after* the fast INT8 matmul.
    Complex LoRA types (LoHA, LoKR, DoRA) fall back to the standard path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_lora"
    CATEGORY = "loaders/ltx"
    DESCRIPTION = (
        "Apply a LoRA to an INT8-quantized model using the additive path. "
        "Skips the dequant-requant roundtrip for simple LoRA types."
    )

    def apply_lora(
        self,
        model,
        lora_name: str,
        strength_model: float,
        clip=None,
        strength_clip: float = 1.0,
    ):
        if strength_model == 0.0 and strength_clip == 0.0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Apply LoRA through standard ComfyUI path first
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_sd,
            strength_model,
            strength_clip if clip is not None else 0.0,
        )

        # Then convert simple LoRA patches on Int8Linear to additive ops
        if strength_model != 0.0:
            _register_lora_additive(new_model)

        return (new_model, new_clip)
