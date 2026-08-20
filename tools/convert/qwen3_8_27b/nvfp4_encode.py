"""Local NVFP4 weight encoder for the fuller Qwen3.8-27B NVFP4 artifact.

Encoder profile ``NVFP4_MAXABS_DIVISOR_RNE_V1``: for one logical BF16 parent
``W [N,K]`` with ``K % 16 == 0``, all arithmetic round-to-nearest-even:

    amax = max(|W|)
    d_w  = binary32(2688 / amax)         # 2688 = 6 * 448; d_w = 1 when amax = 0
    y    = binary32(W * d_w)
    per K-group g of 16:
        ab = max(|y| in g)
        s  = E4M3FN(min(ab / 6, 448))    # +0 when ab = 0 or the cast rounds to 0
        q  = E2M1(y / decode(s))         # ties-to-even, saturating to +-6; s = 0 -> q = +0

The stored weight decodes exactly as the registered NVFP4 contract:
``W_hat = decode_e2m1(q) * decode_e4m3fn(s) / d_w``. Site-level input divisors
``d_x`` use the same orientation (``d_x = binary32(2688 / max|site input|)``) so
one block scale ``E4M3FN(d_x * max_abs(block) / 6)`` fills the A4 range.

This module is the bit-level oracle owner for the E2M1 and E4M3FN words it
emits; every produced payload is round-trip verified against
``tools.artifact.layouts`` word decoding before it is written.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from tools.artifact.layouts import decode_nvfp4_words, encode_nvfp4


ENCODER_PROFILE = "NVFP4_MAXABS_DIVISOR_RNE_V1"

# 6 * 448: the largest product of an E2M1 magnitude and an E4M3FN magnitude.
_FULL_RANGE = 2688.0

# Positive E2M1 magnitudes and the RNE tie boundaries between them. A tie at a
# boundary selects the neighbor whose retained significand bit is even.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_UP = (False, True, False, True, False, True, False)


def _e2m1_decode_table() -> torch.Tensor:
    values = []
    for code in range(16):
        magnitude = _E2M1_MAGNITUDES[code & 7]
        values.append(-magnitude if code & 8 else magnitude)
    return torch.tensor(values, dtype=torch.float32)


def _e4m3fn_decode_table(device: torch.device) -> torch.Tensor:
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    bits = codes.to(torch.int32)
    sign = torch.where((bits & 0x80) != 0, -1.0, 1.0)
    exponent = (bits >> 3) & 0xF
    fraction = bits & 0x7
    normal = (exponent >= 1) & (exponent <= 15)
    normal_or_special = (exponent == 15) & (fraction < 7)
    value = torch.where(
        normal | normal_or_special,
        sign * (1.0 + fraction / 8.0) * torch.pow(
            2.0, (exponent - 7).to(torch.float32)
        ),
        torch.where(
            (exponent == 0) & (fraction != 0),
            sign * fraction * 2.0**-9,
            torch.zeros_like(sign),
        ),
    )
    # e == 15, m == 7 is NaN; it is never produced and decodes to NaN here.
    nan_mask = (exponent == 15) & (fraction == 7)
    return torch.where(nan_mask, torch.full_like(value, float("nan")), value)


def e2m1_rne_codes(values: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to saturating E2M1 codes (0..15, sign bit 3).

    Ties at ``0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`` select the neighbor with
    the even retained significand bit; magnitudes above 5 saturate at 6; signed
    zero keeps its sign.
    """

    sign_bits = torch.where(
        torch.signbit(values), torch.tensor(8, dtype=torch.uint8, device=values.device),
        torch.tensor(0, dtype=torch.uint8, device=values.device),
    )
    magnitude = values.abs()
    code = torch.zeros_like(magnitude, dtype=torch.long)
    for index, boundary in enumerate(_E2M1_BOUNDARIES):
        step_up = magnitude > boundary
        tie_up = (magnitude == boundary) & _E2M1_TIE_UP[index]
        code = torch.where(step_up | tie_up, code + 1, code)
    nibble = torch.gather(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=values.device),
        0,
        code.reshape(-1),
    ).reshape(code.shape)
    return nibble | sign_bits


def _pack_e2m1_pairs(codes: torch.Tensor) -> torch.Tensor:
    """Pack adjacent K codes into bytes: low nibble is the smaller K."""

    pairs = codes.reshape(codes.shape[0], -1, 2)
    return (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    packed_codes: torch.Tensor  # uint8 [N, K/2]
    natural_scales: torch.Tensor  # uint8 [N, K/16]
    weight_divisor: float  # positive finite FP32 value


def quantize_nvfp4(
    weight: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> Nvfp4Words:
    """Quantize one logical BF16/FP32 ``[N,K]`` parent under the profile above."""

    if weight.dim() != 2 or weight.shape[1] % 16 != 0:
        raise ValueError("NVFP4 parents require rank 2 with K divisible by 16")
    target = torch.device(device)
    source = weight.detach().to(device=target, dtype=torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError("NVFP4 source contains NaN or infinity")

    rows, columns = source.shape
    amax = source.abs().amax().item()
    divisor = 1.0 if amax == 0.0 else float(torch.tensor(_FULL_RANGE / amax).item())
    divisor32 = struct.unpack("<f", struct.pack("<f", divisor))[0]
    if divisor32 <= 0.0 or divisor32 != divisor32:
        raise ValueError("NVFP4 weight divisor is not finite and positive")
    divisor_word = torch.tensor(divisor32, dtype=torch.float32, device=target)

    scaled = source * divisor_word
    blocks = scaled.reshape(rows, columns // 16, 16)
    block_amax = blocks.abs().amax(dim=2)
    scale_target = (block_amax / 6.0).clamp(max=448.0)
    scales = scale_target.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_values = scales.view(torch.float8_e4m3fn).to(torch.float32)

    nonzero = scale_values > 0
    ratio = torch.where(
        nonzero.unsqueeze(-1),
        blocks / torch.where(nonzero, scale_values, torch.ones_like(scale_values)).unsqueeze(-1),
        torch.zeros_like(blocks),
    )
    codes = e2m1_rne_codes(ratio)
    return Nvfp4Words(
        packed_codes=_pack_e2m1_pairs(codes.reshape(rows, columns)),
        natural_scales=scales.reshape(rows, columns // 16).contiguous(),
        weight_divisor=divisor32,
    )


def dequantize_nvfp4(
    words: Nvfp4Words,
    *,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Reconstruct represented values ``E2M1 * E4M3FN / d_w`` in FP32."""

    target = torch.device(device)
    rows, groups = words.natural_scales.shape
    columns = groups * 16
    low = (words.packed_codes.to(torch.int32) & 0xF).to(target)
    high = (words.packed_codes.to(torch.int32) >> 4).to(target)
    decode_table = _e2m1_decode_table().to(target)
    codes = torch.stack((low, high), dim=2).reshape(rows, columns)
    values = decode_table[codes.long()]
    scale_values = (
        words.natural_scales.to(target).view(torch.float8_e4m3fn).to(torch.float32)
    )
    return (
        values.reshape(rows, groups, 16)
        * scale_values.unsqueeze(-1)
        / words.weight_divisor
    ).reshape(rows, columns)


def relative_frobenius_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> float:
    left = original.detach().cpu().double()
    right = reconstructed.detach().cpu().double()
    difference = left - right
    denominator = left.norm()
    if denominator == 0.0:
        return 0.0 if difference.norm() == 0.0 else float("inf")
    return (difference.norm() / denominator).item()


def encode_nvfp4_parent(
    weight: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
) -> bytes:
    """Quantize one parent and encode its verified ``blockscale-k16-m128x4-v1`` payload."""

    words = quantize_nvfp4(weight, device=device)
    shape = tuple(weight.shape)
    payload = encode_nvfp4(
        words.packed_codes.cpu(),
        words.natural_scales.cpu(),
        struct.pack("<f", words.weight_divisor),
        shape,
    )
    packed, scales, divisor = decode_nvfp4_words(payload, shape)
    if (
        not torch.equal(packed, words.packed_codes.cpu())
        or not torch.equal(scales, words.natural_scales.cpu())
        or abs(divisor.item() - words.weight_divisor) != 0.0
    ):
        raise RuntimeError("NVFP4 layout word verification failed")
    return payload


def self_test(device: str | torch.device = "cuda") -> dict[str, float]:
    """Verify the codecs against exhaustive decode tables and known ties."""

    target = torch.device(device)
    failures: list[str] = []

    decode_table = _e2m1_decode_table().to(target)
    codes = torch.arange(16, dtype=torch.uint8, device=target)
    signs = torch.where(codes >= 8, -1.0, 1.0)
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, device=target).repeat(2)
    expected = signs * magnitudes
    if not torch.allclose(decode_table, expected, atol=0, rtol=0):
        failures.append("e2m1 decode table")

    grid = torch.tensor(
        [0.0, 0.24, 0.25, 0.26, 0.5, 0.74, 0.75, 0.76, 1.0, 1.24, 1.25, 1.26,
         1.5, 1.74, 1.75, 1.76, 2.0, 2.49, 2.5, 2.51, 3.0, 3.49, 3.5, 3.51,
         4.0, 4.99, 5.0, 5.01, 5.9, 6.0, 7.0, 100.0],
        device=target,
    )
    expected_codes = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
        dtype=torch.uint8,
        device=target,
    )
    actual = e2m1_rne_codes(grid)
    if not torch.equal(actual, expected_codes):
        failures.append(f"e2m1 rne grid: {actual.tolist()}")

    negative = e2m1_rne_codes(-grid)
    if not torch.equal((negative & 7), expected_codes) or not bool(
        ((negative & 8) != 0).all()
    ):
        failures.append("e2m1 negative sign handling")

    fp8_decode = _e4m3fn_decode_table(target)
    sample = torch.tensor(
        [0x00, 0x01, 0x38, 0x40, 0x7E, 0x80],
        dtype=torch.uint8,
        device=target,
    )
    expected_fp8 = torch.tensor(
        [0.0, 2.0**-9, 1.0, 2.0, 448.0, -0.0], device=target
    )
    if not torch.allclose(fp8_decode[sample.long()], expected_fp8, atol=0, rtol=0):
        failures.append("e4m3fn decode table")

    cast_probe = torch.tensor(
        [0.0, 1.0, 1.29, 1.5, 448.0, 447.9, 2.0**-10, 2.0**-9],
        device=target,
    )
    cast_words = cast_probe.to(torch.float8_e4m3fn).view(torch.uint8)
    if torch.equal(
        cast_words,
        torch.tensor([0x00, 0x38, 0x3A, 0x3C, 0x7E, 0x7E, 0x00, 0x01],
                     dtype=torch.uint8, device=target),
    ) is False:
        failures.append(f"e4m3fn cast words: {cast_words.tolist()}")

    if failures:
        raise AssertionError("nvfp4 encoder self-test failed: " + "; ".join(failures))
    return {"checked": 1.0}


__all__ = [
    "ENCODER_PROFILE",
    "Nvfp4Words",
    "dequantize_nvfp4",
    "encode_nvfp4_parent",
    "e2m1_rne_codes",
    "quantize_nvfp4",
    "relative_frobenius_error",
    "self_test",
]
