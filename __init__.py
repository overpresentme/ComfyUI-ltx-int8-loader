"""LTX-Video INT8 custom nodes for ComfyUI.

Based on ltx_standalone's INT8 quantization approach.
  - INT8 tensorwise checkpoint loading (pre-quantized & on-the-fly)
  - Additive LoRA path for Int8Linear layers (no dequant-requant roundtrip)

Source: https://github.com/BobJohnson24/ComfyUI-INT8-Fast (AGPL-3.0)
Modifications: ltx_standalone project (additive LoRA, LTX2 exclusion list)
"""

try:
    from .nodes import LTXInt8CheckpointLoader, LTXInt8AdditiveLoRA

    NODE_CLASS_MAPPINGS = {
        "LTXInt8CheckpointLoader": LTXInt8CheckpointLoader,
        "LTXInt8AdditiveLoRA": LTXInt8AdditiveLoRA,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LTXInt8CheckpointLoader": "LTX2 Checkpoint Loader (INT8)",
        "LTXInt8AdditiveLoRA": "LTX2 LoRA Loader (INT8 Additive)",
    }

except Exception as _e:
    import logging
    logging.getLogger(__name__).error("ltx-int8: failed to load nodes: %s", _e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
