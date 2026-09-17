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
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

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
  # cl (host C++): add the flags directly, scoped so nvcc never sees raw /Zc
  # or /utf-8 tokens.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  )
  # nvcc (CUDA language): the CUDA MSBuild target's Xcompiler property is the
  # host-flag channel itself, so the flags go in raw — a literal -Xcompiler
  # token would be passed to cl as a bogus option.
  list(APPEND CMAKE_CUDA_FLAGS /Zc:preprocessor)
  list(APPEND CMAKE_CUDA_FLAGS /utf-8)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
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
