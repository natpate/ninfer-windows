"""Build the fuller Qwen3.8-27B NVFP4 artifact from its three source roles.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4full \
      --model /path/to/Qwen3.8-27B \
      --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
      --calibration out/qwen3_8_27b_nvfp4full_calibration.json \
      --out out/qwen3_8_27b_nvfp4full.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import encode_direct, encode_nvfp4
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import inventory_nvfp4full as inventory
from . import nvfp4_encode
from . import recipe_nvfp4full as recipe


RECIPE_ID = "qwen3_8_27b_nvfp4full-v1"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4full.ninfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    base_dir: Path
    quantized_dir: Path
    calibration_path: Path
    config_summary: dict[str, object]
    base_source: object
    quantized_dtype_counts: dict[str, int]
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan
    divisor_table: recipe.LocalDivisorTable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(f"{model_dir}: safetensors shard set does not match the index")
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()
    nvfp4_encode.self_test()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    base_dir: str | Path,
    quantized_dir: str | Path,
    calibration_path: str | Path,
) -> ConversionPreflight:
    base = Path(base_dir)
    quantized = Path(quantized_dir)
    calibration = Path(calibration_path)
    _validate_index(base)
    _validate_index(quantized)

    base_config = family_conversion.load_json(base / "config.json")
    if base_config.get("quantization_config") is not None:
        raise ValueError("official source must not declare quantization_config")
    base_summary = family_config.validate_config(base_config)
    quantized_summary = family_config.validate_config(
        family_conversion.load_json(quantized / "config.json")
    )
    if base_summary != quantized_summary:
        raise ValueError("official and quantized source model configs do not match")
    preflight_inventory()

    with ShardReader(quantized) as quantized_reader:
        quantized_dtype_counts = recipe.preflight_quantized_metadata(
            quantized_reader
        )
    divisor_table = recipe.LocalDivisorTable(calibration)
    from tools.convert.qwen3_6.common import recipe as family_recipe

    with ShardReader(base) as base_reader:
        base_source = family_recipe.preflight_source_reader(
            base_reader,
            (
                *recipe.BASE_DIRECT_RECIPES,
                *recipe.OFFICIAL_RECIPES_BY_NAME.values(),
            ),
        )

    resources = base_convert.load_resources(base)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, base)
    return ConversionPreflight(
        base_dir=base,
        quantized_dir=quantized,
        calibration_path=calibration,
        config_summary=base_summary,
        base_source=base_source,
        quantized_dtype_counts=quantized_dtype_counts,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
        divisor_table=divisor_table,
    )


def _encode_source_nvfp4_weight(
    spec: inventory.TensorSpec, reader: ShardReader
) -> bytes:
    selected = recipe.SOURCE_NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_source_nvfp4_weight(
        selected, reader
    )
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _encode_local_nvfp4_weight(
    spec: inventory.TensorSpec, reader: ShardReader, device: torch.device
) -> tuple[bytes, float, float]:
    selected = recipe.LOCAL_NVFP4_WEIGHTS_BY_NAME[spec.name]
    parent = recipe.materialize_bf16_parent(selected.parts, reader)
    if tuple(parent.shape) != spec.shape or parent.dtype != torch.bfloat16:
        raise ValueError(f"{spec.name}: materialized parent signature mismatch")
    words = nvfp4_encode.quantize_nvfp4(parent, device=device)
    payload = encode_nvfp4(
        words.packed_codes.cpu(),
        words.natural_scales.cpu(),
        struct.pack("<f", words.weight_divisor),
        spec.shape,
    )
    error = nvfp4_encode.relative_frobenius_error(
        parent, nvfp4_encode.dequantize_nvfp4(words)
    )
    del parent
    return payload, float(words.weight_divisor), error


def _encode_bf16_exception(
    spec: inventory.TensorSpec, reader: ShardReader
) -> bytes:
    selected = recipe.BF16_WEIGHTS_BY_NAME[spec.name]
    parent = recipe.materialize_bf16_parent(selected.parts, reader)
    if tuple(parent.shape) != spec.shape or parent.dtype != torch.bfloat16:
        raise ValueError(f"{spec.name}: materialized BF16 parent signature mismatch")
    return encode_direct(parent, inventory.BF16)


def _materialize_base_direct(spec: inventory.TensorSpec, reader: ShardReader) -> bytes:
    tensor = family_recipe_materialize(
        recipe.BASE_DIRECT_BY_NAME[spec.name], reader
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return encode_direct(tensor, spec.format)


def family_recipe_materialize(
    tensor_recipe, reader: ShardReader, derived=None
) -> torch.Tensor:
    from tools.convert.qwen3_6.common import recipe as family_recipe

    return family_recipe.materialize_recipe(tensor_recipe, reader, derived)


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
    device: torch.device,
) -> bytes:
    tensor = family_recipe_materialize(
        recipe.OFFICIAL_RECIPES_BY_NAME[spec.name], reader, dict(derived)
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    payload = family_conversion.encode_tensor_payload(tensor, spec, device)
    del tensor
    return payload


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    quantization: Mapping[str, object],
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.base_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.base_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "base": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.base_dir.resolve()),
        },
        "quantized": {
            "repository": recipe.QUANTIZED_REPOSITORY,
            "revision": recipe.QUANTIZED_REVISION,
            "model_path": str(preflight.quantized_dir.resolve()),
            "note": (
                "document-pinned revision 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 "
                "is no longer reachable upstream; this reachable main revision "
                "carries the structurally validated mixed-precision allocation"
            ),
        },
        "calibration": {
            "path": str(preflight.calibration_path.resolve()),
            **preflight.divisor_table.provenance,
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["local_nvfp4"] = quantization
    return report


def convert(
    base_dir: str | Path,
    quantized_dir: str | Path,
    calibration_path: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed three-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"nvfp4full converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_dir, quantized_dir, calibration_path)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.SOURCE_NVFP4_WEIGHT_RECIPES)} source and "
        f"{len(recipe.LOCAL_NVFP4_WEIGHT_RECIPES)} local NVFP4 parents, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    quantization_report: dict[str, object] = {
        "encoder_profile": recipe.LOCAL_ENCODER_PROFILE,
        "parents": {},
    }
    with ShardReader(preflight.base_dir) as base_reader, ShardReader(
        preflight.quantized_dir
    ) as quantized_reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in recipe.SOURCE_NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_source_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.LOCAL_NVFP4_WEIGHTS_BY_NAME:
                    payload, divisor, error = _encode_local_nvfp4_weight(
                        spec, base_reader, resolved_device
                    )
                    quantization_report["parents"][spec.name] = {
                        "weight_scale_divisor": divisor,
                        "relative_frobenius_error": error,
                    }
                elif spec.name in recipe.SOURCE_INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_source_input_divisor(
                        recipe.SOURCE_INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.LOCAL_INPUT_DIVISORS_BY_NAME:
                    scalar = preflight.divisor_table.value(spec.name)
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.BF16_WEIGHTS_BY_NAME:
                    payload = _encode_bf16_exception(spec, base_reader)
                elif spec.name in recipe.BASE_DIRECT_BY_NAME:
                    payload = _materialize_base_direct(spec, base_reader)
                else:
                    payload = _materialize_official(
                        spec, base_reader, derived, resolved_device
                    )
                writer.write(spec.name, payload)
                del payload
                if index % 64 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(
                        f"[{index}/{len(inventory.OBJECT_SPECS)}] objects written",
                        flush=True,
                    )

    errors = [
        entry["relative_frobenius_error"]
        for entry in quantization_report["parents"].values()
    ]
    quantization_report["relative_frobenius_error_max"] = max(errors)
    quantization_report["relative_frobenius_error_mean"] = sum(errors) / len(errors)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(base_dir),
        "quantized_model": str(quantized_dir),
        "calibration": str(calibration_path),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        quantization=quantization_report,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quantized_model,
        arguments.calibration,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
