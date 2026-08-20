"""Closed multi-source recipe for the fuller Qwen3.8-27B NVFP4 artifact.

Three fixed source roles:

- the official BF16 base supplies every locally quantized NVFP4 parent, every
  BF16 exception parent, all direct tensors, both W8 endpoints, and the draft
  head, MTP, and Vision components;
- the quantized Text source supplies the 112 MLP NVFP4 parents (layers 0..55)
  word-for-word together with their weight and input divisors;
- the fixed calibration JSON supplies the 135 locally measured site input
  divisors.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import inventory_nvfp4full as inventory
from . import nvfp4_encode


BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# The document-pinned revision 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 was
# force-pushed out of the upstream repository. 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
# is the reachable `main` revision carrying the identical quantized allocation;
# preflight validates that allocation structurally before anything is copied.
QUANTIZED_REPOSITORY = "unsloth/Qwen3.8-27B-NVFP4"
QUANTIZED_REVISION = "7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
LOCAL_ENCODER_PROFILE = nvfp4_encode.ENCODER_PROFILE


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class SourceNvfp4WeightRecipe:
    """NVFP4 parent whose words are copied from the quantized source."""

    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


@dataclass(frozen=True, slots=True)
class LocalNvfp4WeightRecipe:
    """NVFP4 parent quantized from the official BF16 base at conversion time."""

    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]


@dataclass(frozen=True, slots=True)
class LocalInputDivisorRecipe:
    object_name: str
    weight_name: str


@dataclass(frozen=True, slots=True)
class SourceInputDivisorRecipe:
    object_name: str
    sources: tuple[MatrixSource, ...]
    weight_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Bf16WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _object_layer(object_name: str) -> int:
    return int(object_name.split("/")[2])


def _build_matrix_recipes() -> tuple[
    tuple[SourceNvfp4WeightRecipe, ...],
    tuple[LocalNvfp4WeightRecipe, ...],
    tuple[SourceInputDivisorRecipe, ...],
    tuple[LocalInputDivisorRecipe, ...],
    tuple[Bf16WeightRecipe, ...],
    tuple[tuple[MatrixSource, ...], ...],
]:
    source_weights: list[SourceNvfp4WeightRecipe] = []
    local_weights: list[LocalNvfp4WeightRecipe] = []
    source_divisors: list[SourceInputDivisorRecipe] = []
    local_divisors: list[LocalInputDivisorRecipe] = []
    bf16_weights: list[Bf16WeightRecipe] = []
    divisor_groups: list[tuple[MatrixSource, ...]] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            qkgv_parts = (
                _q_part(query, False),
                _all(key),
                _q_part(query, True),
                _all(value),
            )
            qkgv_name = object_prefix + "attention/query_key_gate_value"
            if layer in inventory.EARLY_ATTENTION_INPUT_LAYERS:
                bf16_weights.append(
                    Bf16WeightRecipe(qkgv_name, (14336, 5120), qkgv_parts)
                )
            else:
                local_weights.append(
                    LocalNvfp4WeightRecipe(qkgv_name, (14336, 5120), qkgv_parts)
                )
                local_divisors.append(
                    LocalInputDivisorRecipe(
                        object_prefix
                        + "attention/input_projection/input_scale_divisor",
                        qkgv_name,
                    )
                )
            output_name = object_prefix + "attention/output"
            if layer in inventory.BF16_ATTENTION_OUTPUT_LAYERS:
                bf16_weights.append(
                    Bf16WeightRecipe(output_name, output.shape, (_all(output),))
                )
            else:
                local_weights.append(
                    LocalNvfp4WeightRecipe(output_name, output.shape, (_all(output),))
                )
                local_divisors.append(
                    LocalInputDivisorRecipe(
                        object_prefix
                        + "attention/output_projection/input_scale_divisor",
                        output_name,
                    )
                )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(source_prefix + "linear_attn.out_proj", 5120, 6144)
            qkvz_name = object_prefix + "gdn/query_key_value_z"
            qkvz_parts = (_all(query_key_value), _all(z))
            local_weights.append(
                LocalNvfp4WeightRecipe(qkvz_name, (16384, 5120), qkvz_parts)
            )
            local_divisors.append(
                LocalInputDivisorRecipe(
                    object_prefix + "gdn/input_projection/input_scale_divisor",
                    qkvz_name,
                )
            )
            output_name = object_prefix + "gdn/output"
            if layer in inventory.BF16_GDN_OUTPUT_LAYERS:
                bf16_weights.append(
                    Bf16WeightRecipe(output_name, output.shape, (_all(output),))
                )
            else:
                local_weights.append(
                    LocalNvfp4WeightRecipe(output_name, output.shape, (_all(output),))
                )
                local_divisors.append(
                    LocalInputDivisorRecipe(
                        object_prefix
                        + "gdn/output_projection/input_scale_divisor",
                        output_name,
                    )
                )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_name = object_prefix + "mlp/gate_up"
        down_name = object_prefix + "mlp/down"
        if layer in inventory.SOURCE_NVFP4_MLP_LAYERS:
            gate_up_sources = (gate, up)
            divisor_groups.append(gate_up_sources)
            source_weights.extend(
                (
                    SourceNvfp4WeightRecipe(
                        gate_up_name,
                        (34816, 5120),
                        (_all(gate), _all(up)),
                        gate_up_sources,
                    ),
                    SourceNvfp4WeightRecipe(
                        down_name, down.shape, (_all(down),), (down,)
                    ),
                )
            )
            source_divisors.extend(
                (
                    SourceInputDivisorRecipe(
                        object_prefix
                        + "mlp/gate_up_projection/input_scale_divisor",
                        gate_up_sources,
                        (gate_up_name,),
                    ),
                    SourceInputDivisorRecipe(
                        object_prefix
                        + "mlp/down_projection/input_scale_divisor",
                        (down,),
                        (down_name,),
                    ),
                )
            )
        else:
            local_weights.extend(
                (
                    LocalNvfp4WeightRecipe(
                        gate_up_name, (34816, 5120), (_all(gate), _all(up))
                    ),
                    LocalNvfp4WeightRecipe(
                        down_name, down.shape, (_all(down),)
                    ),
                )
            )
            local_divisors.extend(
                (
                    LocalInputDivisorRecipe(
                        object_prefix
                        + "mlp/gate_up_projection/input_scale_divisor",
                        gate_up_name,
                    ),
                    LocalInputDivisorRecipe(
                        object_prefix
                        + "mlp/down_projection/input_scale_divisor",
                        down_name,
                    ),
                )
            )

    return (
        tuple(source_weights),
        tuple(local_weights),
        tuple(source_divisors),
        tuple(local_divisors),
        tuple(bf16_weights),
        tuple(divisor_groups),
    )


def _build_base_direct_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = []
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "input_norm",
                family_recipe.source(
                    source_prefix + "input_layernorm.weight", (5120,)
                ),
            )
        )
        if layer in inventory.FULL_ATTENTION_LAYERS:
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/query_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.q_norm.weight", (256,)
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/key_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.k_norm.weight", (256,)
                        ),
                    ),
                )
            )
        else:
            convolution = family_recipe.source(
                source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4)
            )
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_log",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.A_log", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.dt_bias", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/convolution",
                        family_recipe.Transpose(
                            family_recipe.Reshape(
                                family_recipe.Slice(convolution, 1, 0, 1),
                                (10240, 4),
                            ),
                            (1, 0),
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        family_recipe.Concat(
                            (
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_a.weight",
                                    (48, 5120),
                                ),
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_b.weight",
                                    (48, 5120),
                                ),
                            ),
                            0,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/norm",
                        family_recipe.source(
                            source_prefix + "linear_attn.norm.weight", (128,)
                        ),
                    ),
                )
            )
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "post_attention_norm",
                family_recipe.source(
                    source_prefix + "post_attention_layernorm.weight", (5120,)
                ),
            )
        )
    recipes.append(
        family_recipe.TensorRecipe(
            "text/final_norm",
            family_recipe.source("model.language_model.norm.weight", (5120,)),
        )
    )
    return tuple(recipes)


(
    SOURCE_NVFP4_WEIGHT_RECIPES,
    LOCAL_NVFP4_WEIGHT_RECIPES,
    SOURCE_INPUT_DIVISOR_RECIPES,
    LOCAL_INPUT_DIVISOR_RECIPES,
    BF16_WEIGHT_RECIPES,
    WEIGHT_DIVISOR_GROUPS,
) = _build_matrix_recipes()

SOURCE_NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in SOURCE_NVFP4_WEIGHT_RECIPES
}
LOCAL_NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in LOCAL_NVFP4_WEIGHT_RECIPES
}
SOURCE_INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in SOURCE_INPUT_DIVISOR_RECIPES
}
LOCAL_INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in LOCAL_INPUT_DIVISOR_RECIPES
}
BF16_WEIGHTS_BY_NAME = {item.object_name: item for item in BF16_WEIGHT_RECIPES}

NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source for recipe in SOURCE_NVFP4_WEIGHT_RECIPES for part in recipe.parts
    )
)

BASE_DIRECT_RECIPES = _build_base_direct_recipes()
BASE_DIRECT_BY_NAME = {item.object_name: item for item in BASE_DIRECT_RECIPES}
BASE_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in BASE_DIRECT_BY_NAME
)

# Endpoints, draft head, MTP, and Vision reuse the registered official-source
# recipe objects; they are quantized from the BF16 base with the canonical
# grouped-int encoder.
from . import recipe_nvfp4 as registered  # noqa: E402

_OFFICIAL_RECIPES_BY_NAME = dict(registered.OFFICIAL_RECIPES_BY_NAME)
_OFFICIAL_RECIPES_BY_NAME["text/output_head"] = family_recipe.TensorRecipe(
    "text/output_head",
    family_recipe.source("lm_head.weight", (248320, 5120)),
)
OFFICIAL_RECIPES_BY_NAME = _OFFICIAL_RECIPES_BY_NAME
OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name in OFFICIAL_RECIPES_BY_NAME
)


def _validate_parts(
    object_name: str, shape: tuple[int, int], parts: tuple[MatrixPart, ...]
) -> None:
    rows = sum(part.output_rows for part in parts)
    if not parts or (rows, parts[0].source.shape[1]) != shape:
        raise ValueError(f"{object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != shape[1] for part in parts):
        raise ValueError(f"{object_name}: incompatible source K")


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(BASE_DIRECT_RECIPES, BASE_DIRECT_SPECS)
    family_recipe.validate_recipe_coverage(
        tuple(OFFICIAL_RECIPES_BY_NAME[spec.name] for spec in OFFICIAL_TENSOR_SPECS),
        OFFICIAL_TENSOR_SPECS,
    )
    if (
        len(SOURCE_NVFP4_WEIGHT_RECIPES),
        len(LOCAL_NVFP4_WEIGHT_RECIPES),
        len(SOURCE_INPUT_DIVISOR_RECIPES),
        len(LOCAL_INPUT_DIVISOR_RECIPES),
        len(BF16_WEIGHT_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(NVFP4_SOURCES),
        len(BASE_DIRECT_RECIPES),
    ) != (112, 135, 112, 135, 9, 56, 168, 401):
        raise ValueError("Qwen3.8 nvfp4full source recipe is incomplete")

    inventory_nvfp4_names = {spec.name for spec in inventory.NVFP4_TENSOR_SPECS}
    if set(SOURCE_NVFP4_WEIGHTS_BY_NAME) | set(LOCAL_NVFP4_WEIGHTS_BY_NAME) != inventory_nvfp4_names:
        raise ValueError("NVFP4 weight routes do not cover the NVFP4 parents")
    if set(SOURCE_NVFP4_WEIGHTS_BY_NAME) & set(LOCAL_NVFP4_WEIGHTS_BY_NAME):
        raise ValueError("an NVFP4 parent has two materialization routes")
    if set(SOURCE_NVFP4_WEIGHTS_BY_NAME) != {
        spec.name for spec in inventory.SOURCE_NVFP4_WEIGHT_SPECS
    }:
        raise ValueError("source NVFP4 routes do not match the inventory selection")

    divisor_names = {spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS}
    if set(SOURCE_INPUT_DIVISORS_BY_NAME) | set(LOCAL_INPUT_DIVISORS_BY_NAME) != divisor_names:
        raise ValueError("input-divisor routes do not cover the divisor sites")
    if set(SOURCE_INPUT_DIVISORS_BY_NAME) & set(LOCAL_INPUT_DIVISORS_BY_NAME):
        raise ValueError("an input divisor has two materialization routes")

    bound_source = {
        name for site in SOURCE_INPUT_DIVISOR_RECIPES for name in site.weight_names
    }
    if bound_source != set(SOURCE_NVFP4_WEIGHTS_BY_NAME):
        raise ValueError("source input-divisor sites do not bind their parents once")
    bound_local = {site.weight_name for site in LOCAL_INPUT_DIVISOR_RECIPES}
    if bound_local != set(LOCAL_NVFP4_WEIGHTS_BY_NAME):
        raise ValueError("local input-divisor sites do not bind their parents once")

    if set(BF16_WEIGHTS_BY_NAME) != {spec.name for spec in inventory.BF16_EXCEPTION_SPECS}:
        raise ValueError("BF16 exception routes do not match the inventory")

    ownership = (
        set(SOURCE_NVFP4_WEIGHTS_BY_NAME),
        set(LOCAL_NVFP4_WEIGHTS_BY_NAME),
        set(SOURCE_INPUT_DIVISORS_BY_NAME),
        set(LOCAL_INPUT_DIVISORS_BY_NAME),
        set(BF16_WEIGHTS_BY_NAME),
        set(BASE_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME),
    )
    names: set[str] = set()
    for route in ownership:
        if names & route:
            raise ValueError("more than one source route owns an artifact tensor")
        names.update(route)
    if names != {spec.name for spec in inventory.TENSOR_SPECS}:
        missing = {spec.name for spec in inventory.TENSOR_SPECS} - names
        extra = names - {spec.name for spec in inventory.TENSOR_SPECS}
        raise ValueError(f"source routes do not cover the inventory: {sorted(missing)[:2]} {sorted(extra)[:2]}")

    for recipe in SOURCE_NVFP4_WEIGHT_RECIPES:
        _validate_parts(recipe.object_name, recipe.shape, recipe.parts)
    for recipe in LOCAL_NVFP4_WEIGHT_RECIPES:
        _validate_parts(recipe.object_name, recipe.shape, recipe.parts)
    for recipe in BF16_WEIGHT_RECIPES:
        _validate_parts(recipe.object_name, recipe.shape, recipe.parts)


def source_field_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    requirements: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in NVFP4_SOURCES:
        n, k = source.shape
        for suffix, shape, dtype in (
            ("weight_packed", (n, k // 2), "U8"),
            ("weight_scale", (n, k // 16), "F8_E4M3"),
            ("weight_global_scale", (1,), "F32"),
            ("input_global_scale", (1,), "F32"),
        ):
            name = source.field(suffix)
            previous = requirements.setdefault(name, (shape, dtype))
            if previous != (shape, dtype):
                raise ValueError(f"inconsistent source declaration for {name}")
    return requirements


def preflight_quantized_metadata(reader: ShardReader) -> dict[str, int]:
    """Validate that the quantized source carries exactly the MLP allocation."""

    requirements = source_field_requirements()
    missing = set(requirements).difference(reader.names)
    if missing:
        raise ValueError(f"quantized source is missing {sorted(missing)[0]}")
    metadata = reader.metadata(reader.names)
    dtype_counts: dict[str, int] = {}
    for name, (shape, dtype) in requirements.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if dtype_counts != {"U8": 168, "F8_E4M3": 168, "F32": 336}:
        raise ValueError(f"unexpected quantized-source allocation: {dtype_counts}")
    # Every non-MLP linear must exist as a row-scaled FP8 field, proving the
    # checkpoint is the registered mixed-precision allocation.
    fp8_fields = {
        name
        for name, item in metadata.items()
        if item.dtype == "F8_E4M3" and name.endswith(".weight")
    }
    expected_fp8 = 233
    if len(fp8_fields) != expected_fp8:
        raise ValueError(
            f"quantized source FP8 allocation has {len(fp8_fields)} matrices, "
            f"expected {expected_fp8}"
        )
    return dtype_counts


def _word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: divisor must be FP32[1]")
    word = int(tensor.detach().contiguous().cpu().view(torch.int32).item())
    word &= 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return word


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _word(reader.get(source.field(suffix)), source.field(suffix))
        for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} words differ")
    return words[0]


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows) for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def materialize_bf16_parent(
    parts: tuple[MatrixPart, ...], reader: ShardReader
) -> torch.Tensor:
    pieces = [
        _select_rows(reader.get(part.source.field("weight")), part) for part in parts
    ]
    matrix = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
    return matrix


def materialize_source_nvfp4_weight(
    recipe: SourceNvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = source_words.get(part.source)
        if words is None:
            n, k = part.source.shape
            source_packed = reader.get(part.source.field("weight_packed"))
            source_scales = reader.get(part.source.field("weight_scale"))
            if (
                source_packed.dtype != torch.uint8
                or tuple(source_packed.shape) != (n, k // 2)
                or source_scales.dtype != torch.float8_e4m3fn
                or tuple(source_scales.shape) != (n, k // 16)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized NVFP4 source signature mismatch"
                )
            words = (source_packed, source_scales.view(torch.uint8))
            source_words[part.source] = words
        packed_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    divisor = _same_divisor(reader, recipe.divisor_sources, "weight_global_scale")
    if tuple(packed.shape) != (recipe.shape[0], recipe.shape[1] // 2) or tuple(
        scales.shape
    ) != (recipe.shape[0], recipe.shape[1] // 16):
        raise ValueError(
            f"{recipe.object_name}: materialized NVFP4 shape mismatch"
        )
    return packed, scales, struct.pack("<I", divisor)


def materialize_source_input_divisor(
    recipe: SourceInputDivisorRecipe, reader: ShardReader
) -> torch.Tensor:
    word = _same_divisor(reader, recipe.sources, "input_global_scale")
    return torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.float32
    ).reshape(())


class LocalDivisorTable:
    """The fixed calibration result behind every locally measured d_x site."""

    def __init__(self, calibration_path: str | Path):
        document = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        measured = document.get("measured_sites")
        if not isinstance(measured, dict):
            raise ValueError("calibration document has no measured_sites table")
        expected = set(LOCAL_INPUT_DIVISORS_BY_NAME)
        if set(measured) != expected:
            missing = sorted(expected - set(measured))[:2]
            extra = sorted(set(measured) - expected)[:2]
            raise ValueError(f"calibration site set mismatch: {missing} {extra}")
        self._values: dict[str, float] = {}
        for name, entry in measured.items():
            value = entry["input_scale_divisor"]
            word = struct.unpack("<I", struct.pack("<f", float(value)))[0]
            if not valid_positive_fp32_word(word):
                raise ValueError(f"{name}: calibrated input divisor is invalid")
            self._values[name] = float(value)
        self.provenance = {
            key: document.get(key)
            for key in (
                "encoder_profile",
                "divisor_formula",
                "model_path",
                "quantized_model_path",
                "corpus_documents",
                "corpus_tokens",
            )
        }

    def value(self, object_name: str) -> torch.Tensor:
        value = self._values[object_name]
        return torch.frombuffer(
            bytearray(struct.pack("<f", value)), dtype=torch.float32
        ).reshape(())


validate_recipe()


__all__ = [
    "BASE_DIRECT_BY_NAME",
    "BASE_DIRECT_RECIPES",
    "BASE_DIRECT_SPECS",
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "BF16_WEIGHTS_BY_NAME",
    "BF16_WEIGHT_RECIPES",
    "LOCAL_ENCODER_PROFILE",
    "LOCAL_INPUT_DIVISORS_BY_NAME",
    "LOCAL_INPUT_DIVISOR_RECIPES",
    "LOCAL_NVFP4_WEIGHTS_BY_NAME",
    "LOCAL_NVFP4_WEIGHT_RECIPES",
    "LocalDivisorTable",
    "NVFP4_SOURCES",
    "OFFICIAL_RECIPES_BY_NAME",
    "OFFICIAL_TENSOR_SPECS",
    "QUANTIZED_REPOSITORY",
    "QUANTIZED_REVISION",
    "SOURCE_INPUT_DIVISORS_BY_NAME",
    "SOURCE_INPUT_DIVISOR_RECIPES",
    "SOURCE_NVFP4_WEIGHTS_BY_NAME",
    "SOURCE_NVFP4_WEIGHT_RECIPES",
    "materialize_bf16_parent",
    "materialize_source_input_divisor",
    "materialize_source_nvfp4_weight",
    "preflight_quantized_metadata",
    "validate_recipe",
]
