# Building and running NInfer on Windows

This guide covers native Windows 11 x64 source builds of NInfer for an NVIDIA GeForce RTX 5090
(`sm_120a`). Portable releases are linked from the [project README](../README.md); the same
`.ninfer` artifacts, CLI, and HTTP server options apply as on Linux. See the
[CLI guide](cli.md), and [HTTP serving](serving.md) for model downloads, CLI options, and the
serving API.

## Requirements

- Windows 11 x64;
- NVIDIA GeForce RTX 5090 with a driver supporting CUDA 13.1;
- [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-downloads) or newer;
- Visual Studio 2022 with the **Desktop development with C++** workload;
- CMake 3.28 or newer;
- [vcpkg](https://github.com/microsoft/vcpkg); the repository pins the dependency baseline in
  `vcpkg.json`.

The build rejects CUDA architectures other than `120a`, matching the upstream RTX 5090 target.
On Windows, FFmpeg and libcurl come from vcpkg during configure; no system package installation
is required. CUDA 13.1 uses MSVC's conforming preprocessor automatically.

## Installing vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
```

With the vcpkg toolchain file passed to CMake (below), vcpkg installs `curl`, `ffmpeg` (with the
`zlib` feature), and `pkgconf` into the build directory's `vcpkg_installed/` tree, following the
`vcpkg.json` manifest. `vcpkg_installed/` is git-ignored.

## Building from source

From the **x64 Native Tools Command Prompt for VS 2022** (or any shell with the MSVC toolchain
and the CUDA toolkit in `PATH`):

```powershell
git clone https://github.com/natpate/ninfer-windows.git
cd ninfer-windows

cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-windows --config Release --parallel
```

The default configuration builds:

```text
build-windows/apps/Release/ninfer.exe
build-windows/apps/Release/ninfer-serve.exe
```

Tests, benchmarks, and maintainer tools are excluded from the default build, as on Linux.
The release binaries use the FFmpeg, libcurl, zlib, and Winsock DLLs from the
`build-windows/` vcpkg output tree; keep that tree next to the executables or copy the required
DLLs beside them. The CUDA runtime is linked statically, so no `cudart*.dll` is required from the
toolkit.

## Running the CLI

Download an artifact as described in the [project README](../README.md), then:

```powershell
.\build-windows\apps\Release\ninfer.exe models\qwen3_6_27b.ninfer `
  --prompt "Explain prefill and decode in three sentences." `
  --max-context 16384 `
  --max-new 256 `
  --spec mtp --draft-tokens 3 `
  --lm-head-draft
```

Answer content is written to stdout; loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr, exactly as on Linux.

## Running the HTTP server

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_6_27b.ninfer `
  --max-context 16384 `
  --kv-capacity auto `
  --max-concurrency 2 `
  --spec mtp --draft-tokens 3 `
  --lm-head-draft
```

The API is then available at `http://127.0.0.1:8080/v1`. To listen on the network instead of
localhost, pass `--host 0.0.0.0` and allow TCP 8080 through Windows Firewall for the
`ninfer-serve.exe` process.

## DFlash2

Qwen3.8-27B supports DFlash2 with the current `groupwise-int` and `nvfp4` artifacts containing
the complete 66-object `dflash2/` suffix. Older artifacts can still run Text, Vision and MTP
with this build, but selecting DFlash2 reports the missing capability. Portable v0.6.1 predates
DFlash2; build the current source for this backend.

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 `
  --max-concurrency 1 --prefill-chunk 1024 `
  --spec dflash2 --draft-tokens 7 --lm-head-draft
```

Draft counts are startup-fixed in `1..15`; seven is the recommended starting point. Add `--vision`
for image/video input or `--webui` for the bundled WebUI. The Engine reports actual memory
reservations and CUDA Graph readiness during startup.

### Windows validation

On 6 September 2026, the upstream integration through `487f8977` was built and checked on
Windows with MSVC 14.44, CUDA 13.1, driver 616.56 and an RTX 5090. The downloaded Qwen3.8 NVFP4
artifact matched the size and SHA-256 in its model-card manifest.

All three applications built and passed `--help`. Five CPU tests passed: CLI options, server
options, OpenAI schema, Anthropic schema and artifact framing. Fifteen selected GPU suites passed:
candidate selector, linear top-k, dynamic grouped-convolution preparation and fused add,
context-KV materialization, packed-tail RMSNorm, RMSNorm/RoPE, sliding attention, DFlash2 causal
attention and input projection, GDN replay fold, masked-block and ragged-prefix preparation,
speculative round and batched BF16 scatter. The causal-attention subset covers all five KV formats
and CUDA Graph input updates. The independent oracles and criteria are described in
[the test guide](../tests/README.md).

The following real-model configurations passed. Arguments select draft count, Graphs, optimized
head, concurrency, KV type, Vision and extra Device checkpoint slots:

```powershell
cmake --build build-windows --config Release -j --target ninfer_qwen3_8_27b_dflash2_real_test
$env:NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS = 'models\qwen3_8_27b_nvfp4.ninfer'
.\build-windows\tests\Release\ninfer_qwen3_8_27b_dflash2_real_test.exe 7 1 1 1 int8
.\build-windows\tests\Release\ninfer_qwen3_8_27b_dflash2_real_test.exe 15 1 1 2 int8 0 0
.\build-windows\tests\Release\ninfer_qwen3_8_27b_dflash2_real_test.exe 2 0 0 2 bf16 1 0
```

These runs check the fixed ordinary-decoding fixture, penalty handling, repeatable conditional
sampling, partial stops, prefix restoration, ring wraparound and the context-capacity tail.
The Vision configuration also passed image/video requests and recorded two Host-to-Device state
restores. OpenAI non-streaming/SSE content, final usage and `[DONE]`, and an Anthropic Messages
response were checked through HTTP. DFlash2 started at 131,072-token context/KV capacity with
CUDA Graphs and reported 3.04 GiB free after startup in that run.

This is focused Windows validation, not a full test matrix: groupwise-int model execution,
eight concurrent requests and Linux execution were not tested in this campaign.

## Notes and differences from Linux

- Windows uses the Visual Studio generator in the examples above; Ninja Multi-Conf is also
  supported if installed. The build tree layout for multi-config generators is
  `build-windows/apps/Release/`.
- On Windows the CUDA runtime is linked statically (`CUDA::cudart_static`) and the project forces
  a single MSVC runtime library across the CUDA static runtime and the vcpkg dependencies.
- The artifact reader uses memory-mapped files plus unbuffered overlapped reads on Windows and
  `O_DIRECT`/`pread` on POSIX; the 4096-byte alignment contract is identical.
- `ninfer.exe` and `ninfer-serve.exe` are the only required outputs; the Docker path in the
  [project README](../README.md) remains Linux-only.
