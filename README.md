# NInfer-windows

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer-windows is a Windows 11 port of [Neroued/ninfer](https://github.com/Neroued/ninfer), a from-scratch C++/CUDA inference 
engine for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090.
It runs text, image, and video prompts through a local CLI, OpenAI-/Anthropic-compatible HTTP APIs, 
or the included llama.cpp webui. It builds and runs natively on Windows 11 x64. Fork changes should 
also build/run on 64-bit Linux but nothing has been tested there.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 20,437,336,576 bytes (19.03 GiB) | `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 23,719,496,192 bytes (22.09 GiB) | `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c` |
| [Qwen3.8-27B NVFP4F](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer) | `nvfp4full` | `qwen3_8_27b_nvfp4full.ninfer` | 18,324,059,648 bytes (17.07 GiB) | `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

The current Qwen3.8 artifacts include DFlash2 companion weights. Select
`--spec dflash2 --draft-tokens 7 --lm-head-draft` in a current source build; portable v0.6.1
predates this backend. Older Qwen3.8 artifacts remain usable for Text, Vision and MTP in the
current build, but cannot enable DFlash2. See [DFlash2 on Windows](docs/windows.md#dflash2)
for launch and validation commands.

Qwen3.6-27B and Qwen3.8-27B each expose two registered weight profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

## Upstream

NInfer is [Neroued](https://github.com/Neroued)'s project
([Neroued/ninfer](https://github.com/Neroued/ninfer)). This repository is a fork of that
project that adds native Windows support. The engine, model artifacts, API surface, and
published benchmarks are all upstream's work, and the upstream repository remains the
reference implementation (this fork tracks upstream `master` with the additions below).

What this fork adds on top of upstream:

- **Native Windows 11 x64 build and run** — CMake with Visual Studio 2022 (MSVC), with
  [vcpkg](https://github.com/microsoft/vcpkg) resolving FFmpeg, libcurl, and zlib via the
  `vcpkg.json` manifest; the CUDA runtime is statically linked, so the CUDA Toolkit is only
  needed at build time. The Windows compatibility layer is ported from
  [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), without its RTX 3090
  (`sm_86`) retargeting, kernel reschedules, or release packaging.
- **Windows porting of the runtime** — memory-mapped artifact reading with unbuffered
  overlapped I/O (the Windows counterpart of POSIX `O_DIRECT`/`pread`, with the same 4096-byte
  alignment contract), portable console logging and load progress, and portable media
  acquisition for image and video input.
- **MSVC/TMA kernel compatibility** — fixes that let the upstream Blackwell kernels compile
  under MSVC: device-pointer NVFP4 TMA descriptors, the pair-row SwiGLU TMA epilogue, and
  MSVC move-construction details in the target runtime.
- **Stock llama.cpp WebUI** — the HTTP server additionally accepts the stock llama.cpp WebUI's
  API dialect (compatible with the upstream `tools/ui` client), and `ninfer-serve` can serve
  the unmodified WebUI in-process: `--webui` downloads the latest build from the
  [ggml-org/llama-ui](https://huggingface.co/ggml-org/llama-ui) bucket on first start, or
  `--webui-dir DIR` serves an existing local copy.
- **Context window reporting** — `ninfer-serve` advertises the served context ceiling in the
  OpenAI dialect: the objects returned by `/v1/models` and `/v1/models/{id}` carry
  `meta.n_ctx` = the `--max-context` value in force, so clients that auto-detect the context
  window (the stock WebUI, OpenAI-compatible frontends) need no manual configuration.
- **Portable Windows release** — a self-contained zip containing the executables and all
  runtime DLLs; see [Prebuilt Windows release](#prebuilt-windows-release).

Everything else — the Linux build path, the RTX 5090 (`sm_120a`) target, the CUDA 13.1
requirement, and the NVFP4/W4A4 Blackwell execution paths — is unchanged from upstream.

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. [Performance](docs/performance.md) records the exact
benchmark profiles and methodology.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Values are aggregate committed decode throughput and MTP acceptance from complete
intervals whose actual decode batch equaled the configured concurrency.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
in the performance document.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int` | 15,544.3 tok/s | 5,157.1 tok/s | 770.9 tok/s |
| Qwen3.6-27B `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| Qwen3.6-27B `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| Qwen3.8-27B `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| Qwen3.8-27B `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Requirements

NInfer currently requires:

- 64-bit Linux or Windows 11 x64;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler (GCC or Clang on Linux, MSVC from
  Visual Studio 2022 on Windows);
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- `pkg-config` on Linux, or [vcpkg](https://github.com/microsoft/vcpkg) on Windows (the
  repository pins the dependency baseline in `vcpkg.json`);
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. On Linux, NInfer is run from its
source build tree; on Windows, the [prebuilt portable release](#prebuilt-windows-release)
provides the same binaries without a toolchain.

## Prebuilt Windows release

Windows users who would rather not build can use the portable release instead of the build
steps below. The zip is self-contained — executables, all runtime DLLs (FFmpeg,
libcurl, zlib, and the VC++ runtime; the CUDA runtime is statically linked), launcher scripts,
a `models\` folder, a `README.txt`, and `SHA256SUMS`:

1. Download the latest `ninfer-windows-<version>-win64-cuda131.zip` from
   [GitHub Releases](https://github.com/natpate/ninfer-windows/releases). Verify files against
   `SHA256SUMS`, e.g. `Get-FileHash ninfer-serve.exe -Algorithm SHA256`.
2. Extract it anywhere — the launcher scripts use relative paths and work from any location.
3. Download a model into `models\`. Easiest on Windows: run the bundled `download_model.bat`,
   which lists the six published artifacts and downloads the one you pick straight from Hugging
   Face (it follows the redirect, resumes interrupted transfers, and verifies the SHA-256). Or
   download one manually via the Hugging Face CLI, as in [Download a model](#download-a-model).
4. Run the matching launcher, e.g. `.\qwen3_8_27b.bat`. This starts `ninfer-serve` on
   `http://127.0.0.1:8080` (API at `/v1`) and serves the WebUI at the root URL; `--webui`
   downloads the WebUI on first start, so the first run needs an internet connection (later
   runs reuse the local copy).
5. Or run `.\ninfer-serve.exe models\<model>.ninfer [flags]` directly — the options are
   identical to a source build (see [Run the HTTP server](#run-the-http-server)).

The launchers default to a 150,000-token context (`--max-context` / `--default-max-tokens`) to
leave VRAM headroom for the Windows desktop. On the 32 GB RTX 5090, the smaller models
(`qwen3_6_27b`, `qwen3_6_27b_nvfp4`, and `qwen3_8_27b`) can be safely raised to 200,000 when
VRAM is completely free at startup; the two larger models (`qwen3_8_27b_nvfp4` and
`qwen3_6_35b_a3b`) do not fit at 200,000 and should stay at 150,000. Hardware requirements are
unchanged: Windows 11 x64, RTX 5090, and an NVIDIA driver supporting CUDA 13.1.

## Build

### Linux

```bash
git clone https://github.com/natpate/ninfer-windows.git
cd ninfer-windows

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

### Windows

Use Visual Studio 2022 (with MSVC) and vcpkg; the manifest in the repository root pins
`curl`, `ffmpeg`, and `pkgconf`:

```powershell
git clone https://github.com/natpate/ninfer-windows.git
cd ninfer-windows

cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-windows --config Release --parallel
```

The default configuration builds:

```text
build-windows/apps/Release/ninfer.exe
build-windows/apps/Release/ninfer-serve.exe
```

See [the Windows guide](docs/windows.md) for complete setup instructions, vcpkg installation, and
notes on the resulting DLL layout.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.


## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```
```

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or the Qwen3.8-27B NVFP4 full-weight variant:
hf download cometkim/Qwen3.8-27B-nvfp4full-NInfer \
  qwen3_8_27b_nvfp4full.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

## Artifact and startup notes

Current builds accept only version-2 `.ninfer` containers. All six published downloads are version
2. Migration is needed only for Qwen3.6 artifacts downloaded before their version-2 publication:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with the exact older Qwen3.6 NVFP4 or 35B-A3B file. Migration updates container
metadata without rewriting the weight payload.

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct
command diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Windows](docs/windows.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
