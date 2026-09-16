# Weight-only NVFP4 DFlash2 runtime

This standalone runtime branch supports the 34 NVFP4 objects in the Qwen3.8-27B DFlash2 module.
It adds execution/binding support for attention projections, context materialization,
dynamic convolution, linear/SwiGLU projections and the candidate-selector path.
Module execution uses weight-only NVFP4 with BF16 activations, not invented A4 calibration.
It preserves tile-aligned NVFP4 subview scale offsets and admits lookup codebooks
without requiring activation Use metadata that conversion does not produce.

Use a compatible `.ninfer` artifact with a DFlash2 component and select
`--spec dflash2 --draft-tokens 7`; see [DFlash semantics](../maintainer/dflash.md).
This patch contains neither a converter recipe nor Windows, HQ or long-context changes.
Conversion allocation and model-quality qualification remain separate from runtime support.
