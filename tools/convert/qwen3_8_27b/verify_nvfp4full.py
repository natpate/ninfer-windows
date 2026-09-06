"""Verify the fuller Qwen3.8-27B NVFP4 artifact against its three source roles."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from safetensors import safe_open
import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_nvfp4_words,
    encoded_size,
)
from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import convert_nvfp4full as convert
from . import inventory_nvfp4full as inventory
from . import nvfp4_encode
from . import recipe_nvfp4full as recipe
from tools.convert.qwen3_6_27b import verify as base_verify


class VerificationError(ValueError):
    """The artifact does not satisfy the registered nvfp4full contract."""


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    objects: int
    tensors: int
    resources: int
    w8_endpoint_weights: int
    w8_endpoint_rows: int
    w8_endpoint_groups: int
    source_nvfp4_weights: int
    local_nvfp4_weights: int
    local_nvfp4_max_relative_error: float
    bf16_exception_matrices: int
    input_divisors_source: int
    input_divisors_local: int
    payload_bytes: int


def _error(message: str) -> None:
    raise VerificationError(message)


def validate_structure(artifact: Artifact) -> int:
    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        _error(
            f"artifact identity is {artifact.identity!r}, expected "
            f"{expected_identity!r}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _error(
            f"artifact has {len(artifact.objects)} objects, "
            f"expected {len(inventory.OBJECT_SPECS)}"
        )
    cursor = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, inventory.OBJECT_SPECS)
    ):
        if actual.name != expected.name:
            _error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _error(f"{actual.name}: offset {actual.offset}, expected {expected_offset}")
        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _error(f"{actual.name}: expected tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _error(f"{actual.name}: signature {signature} != {registered}")
            if actual.bytes != encoded_size(actual.layout, actual.format, actual.shape):
                _error(f"{actual.name}: encoded byte count is invalid")
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _error(f"{actual.name}: expected resource descriptor")
            if actual.encoding != expected.encoding:
                _error(f"{actual.name}: resource encoding is invalid")
        cursor = actual.offset + actual.bytes
    if dict(formats) != inventory.FORMAT_COUNTS:
        _error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != inventory.LAYOUT_COUNTS:
        _error(f"layout counts are {dict(layouts)}")
    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")
    return payload_bytes


def _verify_resources(artifact: Artifact, base_dir: Path) -> None:
    for spec in inventory.RESOURCE_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj, ResourceObject):
            _error(f"{spec.name}: expected resource")
        source = (base_dir / spec.name.removeprefix("frontend/")).read_bytes()
        if bytes(artifact.payload(obj)) != source:
            _error(f"{spec.name}: resource payload differs from base source")


W8_ENDPOINT_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name in ("text/token_embedding", "text/output_head")
)


def _source_rows(
    reader: ShardReader,
    source: family_recipe.SourceTensor,
    rows: Sequence[int],
) -> torch.Tensor:
    shard = reader.weight_map[source.name]
    with safe_open(
        str(reader.model_dir / shard),
        framework="pt",
        device="cpu",
    ) as handle:
        tensor_slice = handle.get_slice(source.name)
        pieces = [tensor_slice[row : row + 1] for row in rows]
    return torch.cat(pieces, dim=0)


def _verify_w8_endpoints(
    artifact: Artifact,
    reader: ShardReader,
) -> tuple[int, int]:
    verified_rows = 0
    verified_groups = 0
    for spec in W8_ENDPOINT_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj, TensorObject):
            _error(f"{spec.name}: expected tensor")
        expression = recipe.OFFICIAL_RECIPES_BY_NAME[spec.name].expression
        if not isinstance(expression, family_recipe.SourceTensor):
            _error(f"{spec.name}: endpoint source must be one direct BF16 matrix")
        rows = tuple(dict.fromkeys((0, spec.shape[0] // 2, spec.shape[0] - 1)))
        source_rows = _source_rows(reader, expression, rows)
        try:
            verified_groups += base_verify.verify_quantized_rows(
                artifact.payload(obj),
                obj.format,
                obj.shape,
                rows,
                source_rows,
            )
        except base_verify.VerificationError as error:
            _error(f"{spec.name}: {error}")
        verified_rows += len(rows)
    return verified_rows, verified_groups


def _verify_source_nvfp4_weights(
    artifact: Artifact,
    reader: ShardReader,
) -> None:
    for selected in recipe.SOURCE_NVFP4_WEIGHT_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        packed, scales, divisor = recipe.materialize_source_nvfp4_weight(
            selected, reader
        )
        stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(
            artifact.payload(obj), obj.shape
        )
        if not torch.equal(stored_packed, packed):
            _error(f"{obj.name}: packed E2M1 words differ from source")
        if not torch.equal(stored_scales, scales):
            _error(f"{obj.name}: E4M3FN scale words differ from source")
        if bytes(stored_divisor.reshape(1).view(torch.uint8).numpy()) != divisor:
            _error(f"{obj.name}: weight divisor differs from source")


def _verify_local_nvfp4_weights(
    artifact: Artifact,
    reader: ShardReader,
    device: torch.device,
) -> float:
    max_error = 0.0
    for selected in recipe.LOCAL_NVFP4_WEIGHT_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        parent = recipe.materialize_bf16_parent(selected.parts, reader)
        words = nvfp4_encode.quantize_nvfp4(parent, device=device)
        stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(
            artifact.payload(obj), obj.shape
        )
        if not torch.equal(stored_packed, words.packed_codes.cpu()):
            _error(f"{obj.name}: packed E2M1 words differ from the encoder profile")
        if not torch.equal(stored_scales, words.natural_scales.cpu()):
            _error(f"{obj.name}: E4M3FN scale words differ from the encoder profile")
        if stored_divisor.item() != words.weight_divisor:
            _error(f"{obj.name}: weight divisor differs from the encoder profile")
        reconstructed = nvfp4_encode.dequantize_nvfp4(
            nvfp4_encode.Nvfp4Words(
                packed_codes=stored_packed,
                natural_scales=stored_scales,
                weight_divisor=stored_divisor.item(),
            ),
            device=device,
        )
        error = nvfp4_encode.relative_frobenius_error(parent, reconstructed)
        if error > 0.25:
            _error(f"{obj.name}: decode-oracle relative error {error:.4f}")
        max_error = max(max_error, error)
        del parent
    return max_error


def _verify_source_input_divisors(
    artifact: Artifact,
    reader: ShardReader,
) -> None:
    for selected in recipe.SOURCE_INPUT_DIVISOR_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        expected = recipe.materialize_source_input_divisor(selected, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        word = int(stored.view(torch.int32).item()) & 0xFFFFFFFF
        if not valid_positive_fp32_word(word):
            _error(f"{obj.name}: input divisor is not finite and positive")
        if not torch.equal(stored.view(torch.int32), expected.view(torch.int32)):
            _error(f"{obj.name}: input divisor differs from source")


def _verify_local_input_divisors(
    artifact: Artifact,
    divisor_table: recipe.LocalDivisorTable,
) -> None:
    for selected in recipe.LOCAL_INPUT_DIVISOR_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        expected = divisor_table.value(selected.object_name)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        word = int(stored.view(torch.int32).item()) & 0xFFFFFFFF
        if not valid_positive_fp32_word(word):
            _error(f"{obj.name}: input divisor is not finite and positive")
        if not torch.equal(stored.view(torch.int32), expected.view(torch.int32)):
            _error(f"{obj.name}: input divisor differs from calibration")


def _verify_bf16_exception_matrices(
    artifact: Artifact,
    reader: ShardReader,
) -> None:
    for selected in recipe.BF16_WEIGHT_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        expected = recipe.materialize_bf16_parent(selected.parts, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        if not torch.equal(stored.view(torch.int16), expected.view(torch.int16)):
            _error(f"{obj.name}: BF16 words differ from base source")


def verify_artifact(
    artifact: Artifact,
    base_dir: str | Path,
    quantized_dir: str | Path,
    calibration_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> VerificationSummary:
    base = Path(base_dir)
    quantized = Path(quantized_dir)
    payload_bytes = validate_structure(artifact)
    convert.preflight_conversion(base, quantized_dir, calibration_path)
    divisor_table = recipe.LocalDivisorTable(calibration_path)
    _verify_resources(artifact, base)
    with ShardReader(base) as base_reader:
        w8_endpoint_rows, w8_endpoint_groups = _verify_w8_endpoints(
            artifact, base_reader
        )
        _verify_bf16_exception_matrices(artifact, base_reader)
        max_error = _verify_local_nvfp4_weights(
            artifact, base_reader, torch.device(device)
        )
    with ShardReader(quantized) as quantized_reader:
        _verify_source_nvfp4_weights(artifact, quantized_reader)
        _verify_source_input_divisors(artifact, quantized_reader)
    _verify_local_input_divisors(artifact, divisor_table)
    return VerificationSummary(
        objects=len(artifact.objects),
        tensors=len(inventory.TENSOR_SPECS),
        resources=len(inventory.RESOURCE_SPECS),
        w8_endpoint_weights=len(W8_ENDPOINT_SPECS),
        w8_endpoint_rows=w8_endpoint_rows,
        w8_endpoint_groups=w8_endpoint_groups,
        source_nvfp4_weights=len(recipe.SOURCE_NVFP4_WEIGHT_RECIPES),
        local_nvfp4_weights=len(recipe.LOCAL_NVFP4_WEIGHT_RECIPES),
        local_nvfp4_max_relative_error=max_error,
        bf16_exception_matrices=len(recipe.BF16_WEIGHT_RECIPES),
        input_divisors_source=len(recipe.SOURCE_INPUT_DIVISOR_RECIPES),
        input_divisors_local=len(recipe.LOCAL_INPUT_DIVISOR_RECIPES),
        payload_bytes=payload_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    with Artifact.open(arguments.artifact) as artifact:
        summary = verify_artifact(
            artifact,
            arguments.model,
            arguments.quantized_model,
            arguments.calibration,
            device=arguments.device,
        )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
