"""Calibrate NVFP4 site input divisors (d_x) for the fuller Qwen3.8-27B artifact.

The BF16 checkpoint does not fit in device memory beside a resident model, so
this tool streams one decoder layer at a time from the memory-mapped
safetensors shards: a pre-forward hook materializes the layer's parameters on
the GPU, the model's own forward evaluates it, and a post-forward hook evicts
it. Only the token embedding stays resident. Every hooked site records the
maximum absolute represented BF16 activation over the complete fixed corpus:

    d_x = binary32(2688 / amax_site)        # 2688 = 6 * 448

matching the A4 block-scale orientation ``s_x = E4M3FN(d_x * max_abs(block)/6)``.
Sites whose weights come from the quantized source keep their stored divisor
words; this tool measures only locally quantized sites, and additionally
reports the same statistic for unsloth-owned MLP sites as a derivation check.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.calibrate_nvfp4full \
      --model /path/to/Qwen3.8-27B \
      --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
      --out out/qwen3_8_27b_nvfp4full_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from tools.convert.common.safetensors import ShardReader


FULL_RANGE = 2688.0
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
EARLY_ATTENTION_INPUT_LAYERS = (3, 7, 11, 15, 19, 23)
BF16_ATTENTION_OUTPUT_LAYERS = (3, 7)
BF16_GDN_OUTPUT_LAYERS = (4,)
NVFP4_MLP_LAYERS_FROM_SOURCE = tuple(range(56))

# Fixed calibration corpus. The documents are committed with the tool so the
# calibration is deterministic; they cover prose, technical writing, source
# code, and numeric tables.
_CALIBRATION_DOCUMENTS = (
    """The history of numerical weather prediction begins with Lewis Fry Richardson's 1922
attempt to compute atmospheric pressure changes by hand. His forecast, produced by a room
of human calculators working in shifts, took six weeks to advance the atmosphere by six
hours and was wildly inaccurate. Yet the method he described - dividing the atmosphere into
a grid of cells and applying the equations of fluid dynamics to each cell - is precisely
what every modern forecast model does. The difference is that where Richardson needed
weeks of human labor, a contemporary supercomputer performs the same arithmetic in a few
seconds, on a grid a hundred times finer, and assimilates observations from satellites,
aircraft, ocean buoys, and ground stations to initialize the computation. The forecasting
problem is now less about raw computation than about representing small-scale physics:
clouds, turbulence, and convection that fall below the grid resolution.""",
    """Gradient-based optimization underlies nearly all of modern machine learning. Given a
differentiable loss function L parameterized by weights w, training iterates w <- w - eta
* grad L(w), where eta is the learning rate. Stochastic gradient descent estimates the
gradient from minibatches; momentum accumulates a velocity v <- mu * v - eta * grad to
damp oscillations across ill-conditioned valleys; and adaptive methods such as Adam
maintain per-parameter running estimates of the first and second moments, rescaling the
step by an estimate of the gradient's magnitude. Careful initialization matters: drawing
weights from a distribution whose variance matches the fan-in and fan-out of each layer
keeps activation magnitudes stable as signals propagate forward and gradients propagate
backward. Normalization layers - batch norm, layer norm, RMSNorm - remove mean and scale
drift, allowing higher learning rates. Learning-rate schedules decay eta over training:
warmup ramps it up over the first few thousand steps, cosine schedules anneal it toward
zero, and step schedules divide it by a constant at fixed milestones. Regularization by
weight decay, dropout, and data augmentation counteracts overfitting.""",
    """def build_frequency_table(samples: list[int], size: int) -> list[float]:
    counts = [0] * size
    for value in samples:
        if not 0 <= value < size:
            raise ValueError(f"sample out of range: {value}")
        counts[value] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * size
    return [count / total for count in counts]

def entropy(table: list[float]) -> float:
    from math import log2
    return -sum(p * log2(p) for p in table if p > 0.0)

def rolling_checksum(data: bytes, window: int = 32, modulus: int = 16777619) -> int:
    if window <= 0:
        raise ValueError("window must be positive")
    digest = 0
    for i, byte in enumerate(data):
        digest = (digest * 31 + byte) % modulus
        if i >= window:
            digest = (digest - data[i - window] * pow(31, window, modulus)) % modulus
    return digest

class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[float] = []
        self.head = 0

    def push(self, value: float) -> float | None:
        evicted = None
        if len(self.items) == self.capacity:
            evicted = self.items[self.head]
            self.items[self.head] = value
            self.head = (self.head + 1) % self.capacity
        else:
            self.items.append(value)
        return evicted""",
    """Quarterly report extract, all figures in thousands of dollars unless noted.
Revenue: Q1 4,812; Q2 5,377; Q3 5,940; Q4 6,512; full year 22,641, up 18.4 percent
year over year. Cost of revenue: Q1 2,105; Q2 2,318; Q3 2,504; Q4 2,691. Gross margin
improved from 56.2 percent in Q1 to 58.7 percent in Q4. Operating expenses: research and
development 6,204, sales and marketing 4,481, general and administrative 1,902. Operating
income 2,363 for the year, margin 10.4 percent. Net income 1,977 after interest expense
of 214 and tax provision of 172 at an effective rate of 9.6 percent. Cash and equivalents
11,340 at year end; inventory 2,905; accounts receivable 3,661 with days sales
outstanding of 51. Capital expenditure 1,208, of which 744 related to compute capacity.
Headcount ended at 612, up from 545, of which 289 in engineering. Contracted backlog
stood at 9,880 with average duration of 2.3 years.""",
    """Le vent se levait sur la plaine de Beauce balayant les champs de blé presque mûrs.
Les paysans avaient fini la moisson dans le voisinage et les meules dorées attendaient
les chariots. Au loin, on apercevait la flèche de la cathédrale, grise sur le ciel
encore clair, tandis que des nuages épais montaient à l'horizon du côté de la forêt.
La chaleur de la journée s'attardait dans l'air immobile du soir. On entendait
seulement le froissement des feuilles sèches et, de temps en temps, l'aboiement d'un
chien dans quelque ferme éloignée. Elle marcha longtemps sur le chemin creux entre les
haies, ne pensant à rien, regardant la terre de Beauce s'endormir lentement dans la
lumière déclinante. Il lui semblait que cette plaine interminable portait tous les
travaux et toutes les saisons de sa vie, et que chaque sillon connaissait son nom.""",
    """一九四三年秋，昆明的雨季来得比往年早。联大的教室屋顶是铁皮的，雨点打上去，
声音大得讲课的人都听不见自己说话。教授停下来，学生们便合上笔记，听雨。有人
后来回忆说，那几年的知识有一半是在雨声里学的。物价一日三涨，教授们在实验室
外面种菜，在中学兼课，把藏书一册一册地卖出去。可是图书馆晚上依然坐满了人，
一盏油灯下面常常是两个人。物理系的仪器用完了就拆，拆了又装。跑警报的日子，
师生们在郊外的山沟里继续讨论功课，有人带着论文字典，有人带着未完成的实验
记录。那些年在昆明写成的论文，后来散落在世界各地，但其中许多的初稿，是在
铁皮屋顶下、油灯旁边、山沟里的石板上完成的。""",
    """{
  "session": {"id": "a4f7c2e1", "started": "2026-04-03T09:14:22Z", "region": "eu-west-1"},
  "customer": {"tier": "enterprise", "seats": 1240, "renewal": "2027-01-15"},
  "usage": {
    "inference": {"requests": 8421937, "tokens_in": 1120394221, "tokens_out": 402118336},
    "training": {"jobs": 217, "gpu_hours": 18432.5, "checkpoint_gb": 91.3}
  },
  "incidents": [
    {"id": "INC-2201", "severity": 2, "started": "2026-03-28T21:04:11Z",
     "duration_minutes": 43, "affected": ["inference", "dashboard"],
     "cause": "upstream capacity provider failure", "resolved": true},
    {"id": "INC-2207", "severity": 3, "started": "2026-04-01T02:33:57Z",
     "duration_minutes": 11, "affected": ["batch-ingest"], "resolved": true}
  ],
  "capacity": {"gpu_allocated": 512, "gpu_used_peak": 489, "queue_depth_p99": 74}
}""",
    """El camión arrancó al amanecer y tomó la carretera de tierra que subía hacia la
sierra. En la caja, entre sacos de maíz y bidones de agua, iban once personas y
dos gallos. El conductor, que había hecho ese viaje dos veces por semana durante
veinte años, no necesitaba mirar el camino: sabía el nombre de cada curva, de
cada zanja, de cada árbol caído que había que esquivar. A media mañana pararon
en un paraje donde había una cruz de madera y una pileta de agua clara. Allí
comieron tortillas con frijoles y queso, y el más viejo del grupo contó otra vez
la historia del túnel de la mina, la fiebre del oro del año cuarenta y ocho, y
cómo el río cambió de curso una noche de tormenta y se llevó la mitad del pueblo
viejo. Nadie lo interrumpía, aunque todos se la sabían de memoria.""",
    """In compiler design, register allocation is traditionally formulated as a graph
coloring problem: build an interference graph whose nodes are program variables
(live ranges) and whose edges connect variables live at the same program point,
then color the graph with k colors, where k is the number of available registers.
Chaitin's allocator spills a variable when no color can be found, inserting load
and store instructions around the definition and uses, then rebuilding the graph
because spilling changes live ranges. Linear-scan allocation, by contrast, sorts
live intervals by start position and walks them once, keeping an active list and
evicting the interval with the furthest end point when a register is needed -
a single pass with worse register quality but predictable, fast compilation,
which made it the standard choice for just-in-time compilers. SSA form makes
interference sparser: variables defined once interfere only when their live
ranges overlap, and the dominance ordering of definitions enables coalescing of
copy instructions through parallel-copy elimination and lost-copy avoidance.""",
    """# Migration notes: v4 to v5

The v5 release rewrites the storage engine. Read this before upgrading.

**Breaking changes**

1. The manifest format changed from YAML to a length-prefixed binary envelope.
   Run `ninfer migrate --in place/ --out place2/` once; it is idempotent and
   verifies every page checksum before touching the destination.
2. `GET /v4/objects` is removed. Use `GET /v5/objects?cursor=...`; responses
   now return at most 500 names and include a `next_cursor` field.
3. Timestamps are microseconds since epoch (int64), not ISO strings.

**Deprecations**

- `--sort-key` still works but is ignored; v5 always returns canonical order.
- The `x-retention` header moved to the object metadata envelope.

**Operational checklist**

- [ ] Free disk space >= 2.2x the v4 dataset size
- [ ] Backup of the manifest and at least the two most recent snapshots
- [ ] Drain writers (v5 tolerates one stale writer for at most 300 seconds)
- [ ] After migration: `ninfer verify --deep` (about 40 minutes per terabyte)

Rollback: keep the v4 directory until `verify --deep` passes twice in a row.""",
)


def _site_definitions() -> dict[str, str]:
    """Artifact divisor object name -> checkpoint hook parameter path."""

    sites: dict[str, str] = {}
    prefix_ck = "language_model.layers."
    for layer in range(64):
        layer_ck = f"{prefix_ck}{layer}."
        layer_art = f"text/layers/{layer}/"
        is_full = layer in FULL_ATTENTION_LAYERS
        if is_full:
            if layer not in EARLY_ATTENTION_INPUT_LAYERS:
                sites[layer_art + "attention/input_projection/input_scale_divisor"] = (
                    layer_ck + "self_attn.q_proj"
                )
            if layer not in BF16_ATTENTION_OUTPUT_LAYERS:
                sites[layer_art + "attention/output_projection/input_scale_divisor"] = (
                    layer_ck + "self_attn.o_proj"
                )
        else:
            sites[layer_art + "gdn/input_projection/input_scale_divisor"] = (
                layer_ck + "linear_attn.in_proj_qkv"
            )
            if layer not in BF16_GDN_OUTPUT_LAYERS:
                sites[layer_art + "gdn/output_projection/input_scale_divisor"] = (
                    layer_ck + "linear_attn.out_proj"
                )
        if layer not in NVFP4_MLP_LAYERS_FROM_SOURCE:
            sites[layer_art + "mlp/gate_up_projection/input_scale_divisor"] = (
                layer_ck + "mlp.gate_proj"
            )
            sites[layer_art + "mlp/down_projection/input_scale_divisor"] = (
                layer_ck + "mlp.down_proj"
            )
    return sites


# Derivation-check sites: unsloth owns these divisors, we only compare formulas.
_CHECK_SITES = {
    f"text/layers/{l}/mlp/gate_up_projection/input_scale_divisor":
    f"language_model.layers.{l}.mlp.gate_proj"
    for l in NVFP4_MLP_LAYERS_FROM_SOURCE
}


def _read_source_input_divisors(quantized_dir: Path) -> dict[str, float]:
    """Read unsloth FP32 scalars by seek; the single large shard cannot be
    memory-mapped under the host mmap limit."""

    import struct as _struct

    shards = _shard_map(quantized_dir)
    headers: dict[Path, dict] = {}

    def read_scalar(name: str) -> float:
        shard = shards[name]
        if shard not in headers:
            with shard.open("rb") as handle:
                length = _struct.unpack("<Q", handle.read(8))[0]
                header_bytes = handle.read(length)
                headers[shard] = (json.loads(header_bytes), 8 + length)
        directory, base = headers[shard]
        begin = directory[name]["data_offsets"][0]
        with shard.open("rb") as handle:
            handle.seek(base + begin)
            return _struct.unpack("<f", handle.read(4))[0]

    values: dict[str, float] = {}
    for layer in NVFP4_MLP_LAYERS_FROM_SOURCE:
        gate = read_scalar(
            f"model.language_model.layers.{layer}.mlp.gate_proj.input_global_scale"
        )
        up = read_scalar(
            f"model.language_model.layers.{layer}.mlp.up_proj.input_global_scale"
        )
        if gate != up:
            raise ValueError(f"layer {layer}: unsloth gate/up d_x differ")
        values[f"text/layers/{layer}/mlp/gate_up_projection/input_scale_divisor"] = gate
    return values


def _shard_map(model_dir: Path) -> dict[str, Path]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = {}
    for name, shard in index["weight_map"].items():
        shards[name] = model_dir / shard
    return shards


class StreamingLayerLoader:
    """Materializes one decoder layer's parameters on device per forward call."""

    def __init__(self, model, shards: dict[str, Path], device: torch.device):
        self._model = model
        self._shards = shards
        self._device = device
        self._handles = []
        self._open: dict[Path, safe_open] = {}

    def _checkpoint_name(self, param_path: str) -> str:
        return f"model.{param_path}" if not param_path.startswith("model.") else param_path

    def _parameters_of(self, module: torch.nn.Module, prefix: str):
        for child_name, child in module.named_children():
            yield from self._parameters_of(child, f"{prefix}.{child_name}")
        for attr_name, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            yield module, attr_name, f"{prefix}.{attr_name}"

    def _install(self, module: torch.nn.Module, attr_name: str, param_path: str) -> None:
        checkpoint = self._checkpoint_name(param_path)
        shard = self._shards.get(checkpoint)
        if shard is None:
            raise KeyError(f"checkpoint tensor not found: {checkpoint}")
        if shard not in self._open:
            self._open[shard] = safe_open(shard, framework="pt", device="cpu")
        tensor = self._open[shard].get_tensor(checkpoint).to(self._device)
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=False)

    def _evict(self, module: torch.nn.Module, attr_name: str) -> None:
        parameter = module._parameters[attr_name]
        module._parameters[attr_name] = torch.nn.Parameter(
            torch.empty(0, dtype=parameter.dtype, device="meta"),
            requires_grad=False,
        )

    def attach(self) -> None:
        for index, layer in enumerate(self._model.language_model.layers):
            entries = list(self._parameters_of(layer, f"language_model.layers.{index}"))

            def pre_hook(module, args, kwargs, layer_entries=entries):
                for module_, attr_name, param_path in layer_entries:
                    self._install(module_, attr_name, param_path)
                return args, kwargs

            def post_hook(module, args, output, layer_entries=entries):
                for module_, attr_name, _ in layer_entries:
                    self._evict(module_, attr_name)

            self._handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(layer.register_forward_hook(post_hook))

    def load_persistent(self) -> None:
        embedding = self._model.language_model.embed_tokens
        norm = self._model.language_model.norm
        for module, attr_name, param_path in (
            *self._parameters_of(embedding, "language_model.embed_tokens"),
            *self._parameters_of(norm, "language_model.norm"),
        ):
            self._install(module, attr_name, param_path)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._open.clear()


def _amax_hooks(model, sites: dict[str, str], device: torch.device) -> dict[str, float]:
    peaks: dict[str, float] = {name: 0.0 for name in sites}

    def capture(site_name: str):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"{site_name}: hook captured no input tensor")
            value = hidden.detach().abs().amax().item()
            if value > peaks[site_name]:
                peaks[site_name] = value
        return hook

    modules = dict(model.named_modules())
    for site_name, module_path in sites.items():
        module = modules.get(module_path)
        if module is None:
            raise KeyError(f"site module not found: {module_path}")
        module.register_forward_pre_hook(capture(site_name), with_kwargs=True)
    return peaks


def calibrate(
    model_dir: str | Path,
    quantized_dir: str | Path,
    out_path: str | Path,
    *,
    device: str = "cuda",
) -> Path:
    base = Path(model_dir)
    output = Path(out_path)
    target = torch.device(device)
    started = time.perf_counter()

    config = AutoConfig.from_pretrained(base, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base)

    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        from transformers import Qwen3_5Model

        model = Qwen3_5Model(config)
    model.eval()

    shards = _shard_map(base)
    loader = StreamingLayerLoader(model, shards, target)
    loader.load_persistent()
    for module in model.modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device.type == "cpu":
                module._buffers[buffer_name] = buffer.to(target)

    sites = _site_definitions()
    all_sites = {**sites, **_CHECK_SITES}
    peaks = _amax_hooks(model, all_sites, target)

    # One concatenated prefill covers the union corpus in a single stream.
    joined = "\n\n".join(_CALIBRATION_DOCUMENTS)
    chunk_ids = [tokenizer(joined, return_tensors=None)["input_ids"]]

    loader.attach()
    with torch.inference_mode():
        for index, ids in enumerate(chunk_ids, start=1):
            tokens = torch.tensor([ids], dtype=torch.long, device=target)
            begin = time.perf_counter()
            model(tokens, use_cache=False)
            torch.cuda.synchronize()
            print(
                f"chunk {index}/{len(chunk_ids)}: {len(ids)} tokens in "
                f"{time.perf_counter() - begin:.1f}s",
                flush=True,
            )
    loader.close()

    # Unsloth input divisors for the derivation check. The quantized source is
    # one large shard; a full mmap can exceed the host mmap limit, so the four
    # needed bytes per site are read with plain seeks.
    source_divisors: dict[str, float] = _read_source_input_divisors(
        Path(quantized_dir)
    )

    def divisor_for(amax: float) -> float:
        if amax <= 0.0:
            raise ValueError("site amax must be positive over a nonempty corpus")
        return float(torch.tensor(FULL_RANGE / amax).item())

    result = {
        "encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1",
        "divisor_formula": "d_x = binary32(2688 / max|site input| over the fixed corpus)",
        "model_path": str(base.resolve()),
        "quantized_model_path": str(Path(quantized_dir).resolve()),
        "corpus_documents": len(_CALIBRATION_DOCUMENTS),
        "corpus_tokens": sum(len(ids) for ids in chunk_ids),
        "measured_sites": {},
        "derivation_check": {},
    }
    for name in sites:
        amax = peaks[name]
        result["measured_sites"][name] = {
            "amax": amax,
            "input_scale_divisor": divisor_for(amax),
        }
    for name, source_d_x in source_divisors.items():
        implied_amax = FULL_RANGE / source_d_x
        measured = peaks.get(name)
        result["derivation_check"][name] = {
            "source_d_x": source_d_x,
            "implied_amax": implied_amax,
            "measured_amax": measured,
            "ratio_measured_over_implied": (measured / implied_amax) if measured else None,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"calibration complete: {len(sites)} sites, {len(source_divisors)} checks, "
        f"{result['corpus_tokens']} tokens in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    calibrate(arguments.model, arguments.quantized_model, arguments.out,
              device=arguments.device)


if __name__ == "__main__":
    main()
