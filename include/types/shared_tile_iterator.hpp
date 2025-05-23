// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/compute/gemm.hpp"
#include "types/shared.hpp"
#include "types/tile_shape.hpp"

#include <functional>
#include <iostream>
#include <optional>

namespace tilefusion::cell {
namespace tl = tile_layout;
using namespace compute;

namespace {
/// @brief Helper for pretty printing a tile iterator's static shape-related
///        information. This printer works ONLY on the host.
struct STileIteratorPrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "SharedTileItertor {" << std::endl
            << "  ChunkShape = (" << TileIterator::kChunkRows << ", "
            << TileIterator::kChunkCols << "), stripe count = ("
            << TileIterator::sc0 << ", " << TileIterator::sc1 << ")"
            << std::endl
            << "}";
    }
};

struct STileIteratorPrett2yPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "SharedTileItertor2 {" << std::endl
            << "  ChunkShape = (" << TileIterator::kChunkRows << ", "
            << TileIterator::kChunkCols << "), stripe count = ("
            << TileIterator::sc0 << ", " << TileIterator::sc1 << ")"
            << std::endl
            << "}";
    }
};
}  // namespace

/// @brief Type trait to detect if a layout is a BlockMatrxLayout
template <typename Layout>
struct is_block_layout : std::false_type {};

template <typename OuterLayout, typename InnerLayout>
struct is_block_layout<tl::BlockMatrxLayout<OuterLayout, InnerLayout>>
    : std::true_type {};

template <typename Layout>
static constexpr bool is_block_layout_v = is_block_layout<Layout>::value;

/// @brief Helper to create the appropriate sub-tile layout type
template <typename TileLayout, int kChunkRows, int kChunkCols,
          bool IsBlockLayout = is_block_layout_v<TileLayout>>
struct SubTileLayoutCreator;

/// @brief Specialization for simple MatrixLayout
template <typename TileLayout, int kChunkRows, int kChunkCols>
struct SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols, false> {
    static constexpr int kTileRowStride =
        TileLayout::kType == tl::Layout::kRowMajor ? TileLayout::kCols : 1;
    static constexpr int kTileColStride =
        TileLayout::kType == tl::Layout::kRowMajor ? 1 : TileLayout::kRows;

    using type = tl::MatrixLayout<kChunkRows, kChunkCols, kTileRowStride,
                                  kTileColStride>;
};

/// @brief Specialization for BlockMatrxLayout
template <typename TileLayout, int kChunkRows, int kChunkCols>
struct SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols, true> {
    // For block layouts, we need to preserve the block structure
    // The sub-tile should have the same layout type as the original

    // Extract inner layout dimensions for proper sub-tile creation
    static constexpr int kInnerRows = TileLayout::kInnerRows;
    static constexpr int kInnerCols = TileLayout::kInnerCols;

    // Calculate how many complete blocks fit in the chunk
    static constexpr int kChunkTileRows = kChunkRows / kInnerRows;
    static constexpr int kChunkTileCols = kChunkCols / kInnerCols;

    // Create outer layout for the sub-tile
    using OuterLayout =
        std::conditional_t<TileLayout::kType == tl::Layout::kRowMajor,
                           tl::RowMajor<kChunkTileRows, kChunkTileCols>,
                           tl::ColMajor<kChunkTileRows, kChunkTileCols>>;

    // Use the same inner layout as the original
    using InnerLayout = typename TileLayout::InnerLayout;

    using type = tl::BlockMatrxLayout<OuterLayout, InnerLayout>;
};

template <typename TileLayout, int kChunkRows, int kChunkCols>
using SubTileLayout_t =
    typename SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols>::type;

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
/// @param Tile_: The type of the large tile to chunk.
/// @param ChunkShape_: The shape of the smaller tiles into which the large
///                     tile is partitioned (chunk shape).
template <class Tile_, class ChunkShape_>
class STileIterator {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;

    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;

    static constexpr int kChunkRows = dim_size<0, ChunkShape>;
    static constexpr int kChunkCols = dim_size<1, ChunkShape>;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int sc0 = Tile::kRows / kChunkRows;
    static constexpr int sc1 = Tile::kCols / kChunkCols;

    HOST_DEVICE STileIterator() : data_(nullptr) {}

    DEVICE STileIterator(DType* data) : data_(data) {}

    DEVICE STileIterator(const DType* data) : data_(const_cast<DType*>(data)) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.
    DEVICE auto operator()(int i) {
        assert(data_);  // The iterator is not initialized.
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the strip count "
                      "of one of the iterator's dimensions is 1.");

        int x = sc0 == 1 ? 0 : i;
        int y = sc0 == 1 ? i : 0;

        using TileLayout = tl::MatrixLayout<kChunkRows, kChunkCols,
                                            kTileRowStride, kTileColStride>;

        using NewTile =
            SharedTile<DType, TileLayout, Tile::kSwizzled, Tile::SwizzleBytes>;

        // TODO(KuangjuX): hotfix for `offset1` and `offset2`.
        int offset1 = x * (kChunkRows * Tile::kRowStride) +
                      y * kTilePerChunkCol * BaseShape::kCols;
        int offset2 = x * kTilePerChunkRow * BaseShape::kRows +
                      y * (Tile::kColStride * kChunkCols);
        int offset = Tile::kType == tl::Layout::kRowMajor ? offset1 : offset2;

        NewTile tile(data_ + offset, offset);
        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto to_tile() {
        Tile tile(data_);
        return tile;
    }

  private:
    static constexpr int kTilePerRow = Tile::kRows / BaseShape::kRows;
    static constexpr int kTilePerCol = Tile::kCols / BaseShape::kCols;

    static constexpr int kTilePerChunkRow = kChunkRows / BaseShape::kRows;
    static constexpr int kTilePerChunkCol = kChunkCols / BaseShape::kCols;

    // TODO(KuangjuX): hotfix for `kTileRowStride` and `kTileColStride`.
    static constexpr int kTileRowStride =
        Tile::kType == tl::Layout::kRowMajor ? Tile::kCols : 1;

    static constexpr int kTileColStride =
        Tile::kType == tl::Layout::kRowMajor ? 1 : Tile::kRows;

    DType* data_;
};

/// @brief Pretty printer for the static shape information of a TileIterator.
///        Note: This printer function works ONLY on the host.
template <typename TileShape, typename ChunkShape>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator<TileShape, ChunkShape>& itr) {
    STileIteratorPrettyPrinter::print(out, itr);
    return out;
}

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
template <class Tile_, class ChunkShape_>
class STileIterator2 {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;

    static constexpr int kChunkRows = dim_size<0, ChunkShape>;
    static constexpr int kChunkCols = dim_size<1, ChunkShape>;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int sc0 = Tile::kRows / kChunkRows;
    static constexpr int sc1 = Tile::kCols / kChunkCols;

    static_assert(sc0 >= 1 && sc1 >= 1,
                  "The strip count of both dimensions must be >= 1.");

    HOST_DEVICE STileIterator2() : tile_(nullptr), data_(nullptr) {}

    DEVICE STileIterator2(Tile* tile)
        : tile_(tile), data_(const_cast<DType*>(tile->data())) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.
    DEVICE auto operator()(int i) {
        assert(tile_ && data_);  // The iterator is not initialized.
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the strip count "
                      "of one of the iterator's dimensions is 1.");

        int x = sc0 == 1 ? 0 : i;
        int y = sc0 == 1 ? i : 0;

        // Use the dynamic layout creator to get the appropriate sub-tile layout
        using TileLayout =
            SubTileLayout_t<typename Tile::Layout, kChunkRows, kChunkCols>;
        using NewTile = SharedTile<DType, TileLayout>;

        // Calculate offset - STileIterator2 uses simplified calculation for
        // both layout types
        int offset = x * (kChunkRows * tile_->kRowStride) +
                     y * (kChunkCols * tile_->kColStride);

        NewTile tile(data_ + offset, offset);
        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto to_tile() {
        assert(tile_);
        return *tile_;
    }

  private:
    Tile* tile_;
    DType* data_;
};

/// @brief Pretty printer for the static shape information of a TileIterator.
///        Note: This printer function works ONLY on the host.
template <class Tile_, class ChunkShape_>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator2<Tile_, ChunkShape_>& itr) {
    STileIteratorPrett2yPrinter::print(out, itr);
    return out;
}
}  // namespace tilefusion::cell
