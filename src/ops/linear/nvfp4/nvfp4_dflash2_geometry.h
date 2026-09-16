#pragma once
// DFlash2 drafter matrix geometries for the NVFP4-encoded draft module, shared by the
// fused attention-input route and the registered linear shapes.
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

using Nvfp4DFlash2FeatureGeometry  = Nvfp4Geometry<5120, 25600>;
using Nvfp4DFlash2QkvGeometry      = Nvfp4Geometry<6144, 5120>;
using Nvfp4DFlash2AttnOutGeometry  = Nvfp4Geometry<5120, 4096>;
using Nvfp4DFlash2ConvProjGeometry = Nvfp4Geometry<1280, 5120>;
using Nvfp4DFlash2SelectorGeometry = Nvfp4Geometry<256, 5120>;

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;

} // namespace ninfer::ops::detail
