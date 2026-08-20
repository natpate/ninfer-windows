"""Split an oversized quantized-source shard into ~2 GiB shards.

The upstream quantized checkpoint stores all Text tensors in one shard larger
than the host will reliably memory-map while other model shards are resident.
This mechanical repack copies every tensor byte-for-byte into bounded shards
and rewrites ``model.safetensors.index.json``; it changes no values, names,
dtypes, or shapes. The copy path parses the safetensors header and reads tensor
spans with plain file reads, so it never memory-maps the oversized source.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.split_quantized_shards \
      --model /path/to/Qwen3.8-27B-NVFP4 \
      --target-shard-bytes 2147483648
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct

import torch
from safetensors.torch import save_file


TARGET_SHARD_BYTES = 2 * 1024 * 1024 * 1024
OUTPUT_SHARD_PREFIX = "model-split"

_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I64": torch.int64,
    "I32": torch.int32,
}


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return header, 8 + length


def _read_tensor(path: Path, base: int, entry: dict) -> torch.Tensor:
    begin, end = entry["data_offsets"]
    dtype = _DTYPES[entry["dtype"]]
    element_size = torch.empty((), dtype=dtype).element_size()
    count = (end - begin) // max(1, element_size)
    with path.open("rb") as handle:
        handle.seek(base + begin)
        raw = handle.read(end - begin)
    shape = entry.get("shape", [])
    if not shape:
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(())
    elements = 1
    for dim in shape:
        elements *= dim
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)


def split_shards(model_dir: str | Path, target_bytes: int = TARGET_SHARD_BYTES) -> None:
    base = Path(model_dir)
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    new_map: dict[str, str] = {}
    part = 0
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush(total_hint: int | None = None) -> None:
        nonlocal part, current, current_bytes
        if not current:
            return
        part += 1
        total = total_hint if total_hint is not None else part
        name = f"{OUTPUT_SHARD_PREFIX}-{part:05d}-of-{total:05d}.safetensors"
        save_file(current, str(base / name))
        for tensor_name in current:
            new_map[tensor_name] = name
        current = {}
        current_bytes = 0

    for shard in sorted(by_shard):
        if shard.startswith(OUTPUT_SHARD_PREFIX):
            continue
        header, data_base = _read_header(base / shard)
        for name in by_shard[shard]:
            entry = header[name]
            begin, end = entry["data_offsets"]
            size = end - begin
            if current_bytes and current_bytes + size > target_bytes:
                flush()
            current[name] = _read_tensor(base / shard, data_base, entry)
            current_bytes += size
    pending = current
    total_estimate = part + (1 if pending else 0)
    flush(total_hint=total_estimate)
    if part != total_estimate:
        # The final flush disagreed with the estimate; rewrite only its name.
        raise RuntimeError("final shard count estimate failed; rerun the split")
    final_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_map,
    }
    index_path.write_text(json.dumps(final_index, indent=2))
    print(f"split {len(new_map)} tensors into {total} shards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--target-shard-bytes", type=int, default=TARGET_SHARD_BYTES)
    arguments = parser.parse_args()
    split_shards(arguments.model, arguments.target_shard_bytes)


if __name__ == "__main__":
    main()
