// DFlash2 drafter matrices for the NVFP4-encoded draft module: weight-only A16 routes
// (gemv single token, exact small-T families, 32-token chunks for wider extents). The draft
// never quantizes activations, so no A4 route is registered for these shapes.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_dflash2_geometry.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact = Nvfp4SimtSchedule<(Tokens >= 8 && Tokens <= 16) ? 16 : 4, 1, 2, 16, Tokens, 1,
                                Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                                Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C2  = Nvfp4SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                              Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                              Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4  = Nvfp4SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                              Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                              Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C32 = Nvfp4SimtSchedule<4, 1, 2, 16, 16, 1, Nvfp4SimtActivationAccess::TokenPacked,
                              Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                              Nvfp4SimtBlockOrder::TokenTilesContiguous, 3>;
using FullChunk = Nvfp4SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;

template <class Geometry>
Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_nvfp4_gemv<Geometry, Gemv>;
    if (tokens == 32) return launch_nvfp4_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= 5 && tokens <= 28) return select_nvfp4_exact<Geometry, 5, 28, Exact>(tokens);
    if (tokens <= 2) return launch_nvfp4_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return launch_nvfp4_simt<Geometry, 4, C4, false>;
    if (tokens <= 32) return launch_nvfp4_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 DFlash2 A16 chunk exceeds shape capacity");
}

template <class Geometry>
const Nvfp4LinearShape make_dflash2_shape() {
    return {Geometry::kOutputRows, Geometry::kInputRows,
            launch_nvfp4_a16_chunks<32, select_a16<Geometry>>, nullptr,
            [](std::int32_t, std::int32_t) { return false; }};
}

using Feature  = Nvfp4Geometry<5120, 25600>;
using Qkv      = Nvfp4Geometry<6144, 5120>;
using AttnOut  = Nvfp4Geometry<5120, 4096>;
using ConvProj = Nvfp4Geometry<1280, 5120>;
using Selector = Nvfp4Geometry<256, 5120>;

} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Feature  = make_dflash2_shape<Feature>();
const Nvfp4LinearShape kNvfp4DFlash2Qkv      = make_dflash2_shape<Qkv>();
const Nvfp4LinearShape kNvfp4DFlash2AttnOut  = make_dflash2_shape<AttnOut>();
const Nvfp4LinearShape kNvfp4DFlash2ConvProj = make_dflash2_shape<ConvProj>();
const Nvfp4LinearShape kNvfp4DFlash2Selector = make_dflash2_shape<Selector>();

} // namespace ninfer::ops::detail
