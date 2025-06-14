// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/container/tuple.hpp>
#include <cute/int_tuple.hpp>

namespace tilefusion {

template <size_t... Ns>
struct TileShape {
  static constexpr cute::array<size_t, sizeof...(Ns)> shape = {Ns...};

  static constexpr size_t get_numel() {
    size_t product = 1;
    for (size_t n : shape) product *= n;
    return product;
  }

  static constexpr size_t kNumel = get_numel();
};

template <typename TileShape>
inline static constexpr int64_t get_numel = TileShape::kNumel;

template <const size_t I, typename TileShape>
inline static constexpr size_t dim_size = cute::get<I>(TileShape::shape);

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing
}  // namespace tilefusion
