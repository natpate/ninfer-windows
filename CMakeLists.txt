cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build the NInfer test suite" OFF)
option(NINFER_BUILD_BENCHMARKS "Build the NInfer benchmark binaries" OFF)
# Media acquisition, product logging, prompt input and the HTTP serve stack.
# Undefined before this: only a -D flag on the configure command line set it,
# and a cache without it silently dropped the whole product/serve target tree.
option(NINFER_BUILD_PRODUCT_SUPPORT "Build product support (media, logging, prompt input, serve)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # CUDA 13.3's CCCL headers require the standard-conforming preprocessor
  # (their macro state is corrupted under the traditional one: cub's
  # CUB_NAMESPACE_BEGIN expands empty and the CUB headers fail to parse), and
  # spdlog's bundled fmt hard-asserts a /utf-8 execution charset. The CCCL
  # suppression macro is kept so its #error stays quiet even if the
  # preprocessor flag is reverted.
  add_compile_definitions(CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
  # windows.h's min/max macros collide with std::numeric_limits<...>::max()
  # and std::min/max under the standard-conforming preprocessor (the
  # traditional PP's post-'::' macro-skip quirk had masked the collision
  # tree-wide). Define NOMINMAX for every MSVC target; per-file guards
  # (request_log.cpp, context_cost.cpp, media_acquire) stay in place and are
  # now redundant but harmless.
  add_compile_definitions(NOMINMAX)
  # NOTE: /Zc:preprocessor (CUDA 13.3 CC/CUB macro corruption fix) and /utf-8
  # (spdlog fmt requirement) are delivered at build time via the CL
  # environment variable, not CMake flag variables — every CMake channel
  # for them is broken or dead under the CUDA 13.3 MSBuild target:
  #   - CMAKE_CXX_FLAGS / add_compile_options: raw tokens land on the nvcc
  #     command line, and nvcc treats unknown /-form tokens as input files
  #     ("single input file is required" fatal).
  #   - CMAKE_CUDA_FLAGS: items never reach the -Xcompiler value the CUDA
  #     target builds (dead channel); multi-item lists serialize ';'-joined
  #     and cl mangles them (D9002).
  # So: set CL in the build prompt before building:
  #   set "CL=/Zc:preprocessor /utf-8"
  # cl.exe (including the host step nvcc spawns) reads CL on every
  # invocation.
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
