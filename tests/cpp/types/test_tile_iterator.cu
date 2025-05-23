// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>
#include <thrust/host_vector.h>

namespace tilefusion::testing {

using namespace cell;
namespace tl = tile_layout;

namespace {
template <typename Layout, typename ChunkShape, const tl::Layout kType>
struct GTileIteratorTester;

template <typename Layout_, typename ChunkShape>
struct GTileIteratorTester<Layout_, ChunkShape, tl::Layout::kRowMajor> {
    using Element = float;
    using Layout = Layout_;

    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;

    static constexpr int kStride0 = dim_size<0, ChunkShape>;
    static constexpr int kStride1 = dim_size<1, ChunkShape>;

    const int kTileRowStride = kStride0 * Layout::kRowStride;
    const int kTileColStride = kStride1;

    static_assert(kRows % kStride0 == 0, "kRows must be divisible by kStride0");
    static_assert(kCols % kStride1 == 0, "kCols must be divisible by kStride1");

    using Tile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using Iterator = GTileIterator<Tile, ChunkShape>;

    void operator()() {
        int numel = kRows * kCols;
        thrust::host_vector<Element> data(numel);

        Layout layout;
        Element* ptr = data.data();
        int count = 0;
        for (int i = 0; i < kRows; ++i)
            for (int j = 0; j < kCols; ++j) ptr[count++] = layout(i, j);

#if defined(DEBUG_PRINT)
        Tile gtile(ptr);
        gtile.dump_value();
#endif

        EXPECT_EQ(Iterator::sc0, kRows / kStride0);
        EXPECT_EQ(Iterator::sc1, kCols / kStride1);

        Iterator iter(data.data());

        for (int i = 0; i < Iterator::sc0; ++i) {
            for (int j = 0; j < Iterator::sc1; ++j) {
                int start_n = i * kTileRowStride + j * kTileColStride;
                auto tile = iter(i, j);
                for (int m = 0; m < kStride0; ++m) {
                    for (int n = 0; n < kStride1; ++n) {
                        int v1 = int(tile(m, n));
                        int v2 = start_n + m * Layout::kRowStride + n;
                        EXPECT_EQ(v1, v2);
                    }
                }

#if defined(DEBUG_PRINT)
                printf("\nIteration-[%d, %d]:\n", i, j);
                iter(i, j).dump_value();
                printf("\n");
#endif
            }
        }
    }
};

template <typename Layout_, typename ChunkShape>
struct GTileIteratorTester<Layout_, ChunkShape, tl::Layout::kColMajor> {
    using Element = float;
    using Layout = Layout_;

    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;

    static constexpr int kStride0 = dim_size<0, ChunkShape>;
    static constexpr int kStride1 = dim_size<1, ChunkShape>;

    const int kTileRowStride = kStride0;
    const int kTileColStride = kStride1 * Layout::kColStride;

    static_assert(kRows % kStride0 == 0, "kRows must be divisible by kStride0");
    static_assert(kCols % kStride1 == 0, "kCols must be divisible by kStride1");

    using Tile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using Iterator = GTileIterator<Tile, ChunkShape>;

    void operator()() {
        int numel = kRows * kCols;
        thrust::host_vector<Element> data(numel);

        Layout layout;
        Element* ptr = data.data();
        int count = 0;
        for (int i = 0; i < kRows; ++i)
            for (int j = 0; j < kCols; ++j) ptr[count++] = layout(i, j);

#if defined(DEBUG_PRINT)
        Tile gtile(ptr);
        gtile.dump_value();
#endif

        EXPECT_EQ(Iterator::sc0, kRows / kStride0);
        EXPECT_EQ(Iterator::sc1, kCols / kStride1);

        Iterator iter(data.data());

        for (int i = 0; i < Iterator::sc0; ++i) {
            for (int j = 0; j < Iterator::sc1; ++j) {
                int start_n = i * kTileRowStride + j * kTileColStride;

                auto tile = iter(i, j);
                for (int m = 0; m < kStride0; ++m) {
                    for (int n = 0; n < kStride1; ++n) {
                        int v1 = int(tile(m, n));
                        int v2 = start_n + m + n * Layout::kColStride;

                        EXPECT_EQ(v1, v2);
                    }
                }

#if defined(DEBUG_PRINT)
                printf("\nIteration-[%d, %d]:\n", i, j);
                iter(i, j).dump_value();
                printf("\n");
#endif
            }
        }
    }
};

__device__ void init_buf(void* buf, int numel) {
    for (int i = 0; i < numel; ++i) {
        reinterpret_cast<int*>(buf)[i] = i;
    }
}

template <typename Shared, typename SIterator>
__global__ void test_shared_tile_iterator() {
    using DType = typename Shared::DType;
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    DType* buf = reinterpret_cast<DType*>(buf_);

    init_buf(buf, Shared::kNumel);

    Shared s_tile(buf);

    s_tile.dump_value();
}

}  // namespace

// TEST(TestGTileIterator, test_row_major) {
//     using Tester = GTileIteratorTester<tl::RowMajor<4, 9>, TileShape<2, 3>,
//                                        tl::Layout::kRowMajor>;
//     Tester tester;
//     tester();
// }

// TEST(TestGTileIterator, col_major) {
//     using Tester = GTileIteratorTester<tl::ColMajor<4, 9>, TileShape<2, 3>,
//                                        tl::Layout::kColMajor>;
//     Tester tester;
//     tester();
// }

TEST(TestSharedTileIterator, row_major) {
    using InType = __half;
    static constexpr int kRows = 16;
    static constexpr int kCols = 64;

    static constexpr int kChunkRows = 8;
    static constexpr int kChunkCols = 64;

    using SharedLayout = tl::BlockRowMajor<
        tl::RowMajor<kRows, kCols>,
        SwizzledLayout<tl::RowMajor<8, 64>, Swizzle<3, 3, 3>>>;

    using Shared = SharedTile<InType, SharedLayout>;
    using SIterator = STileIterator<Shared, TileShape<kChunkRows, kChunkCols>>;

    LOG(INFO) << std::endl << Shared{} << std::endl;
    LOG(INFO) << std::endl << SIterator{} << std::endl;

    test_shared_tile_iterator<Shared, SIterator><<<1, 1>>>();
}
}  // namespace tilefusion::testing
