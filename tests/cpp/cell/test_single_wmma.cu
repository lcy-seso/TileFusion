// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <glog/logging.h>

namespace tilefusion::testing {
using namespace cell;
using namespace copy;
using namespace compute;
namespace tl = tile_layout;

namespace {
// In this implementation, A and C are interpreted as being laid out in
// row-major, and B is interpreted as being laid out in column-major.
__device__ void naive_gemm(int kM, int kN, int kK,  //
                           const __half* A, const __half* B, float* C) {
  if (!thread0()) return;

  for (int i = 0; i < kM; ++i) {
    for (int j = 0; j < kN; ++j) {
      float s = 0.;
      for (int k = 0; k < kK; ++k) {
        s += __half2float(A[i * kK + k]) * __half2float(B[k + kK * j]);
      }
      C[i * kN + j] = s;
    }
  }
}

__device__ void check_results(const float* hc1, const float* hc2, int numel) {
  for (int i = 0; i < numel; ++i) {
    if (fabs(hc1[i] - hc2[i]) > 1e-3) {
      printf("error: %d, %.4f, %.4f\n", i, hc1[i], hc2[i]);
      printf("test failed!\n");
      return;
    }
  }

  printf("test passed!\n");

#if defined(DEBUG)
  if (thread0()) {
    int cut_off = numel < 128 ? numel : 128;
    printf("\nours:\n");
    printf("%d:\t", 0);
    for (int i = 0; i < cut_off; i++) {
      printf("%.2f, ", hc1[i]);
      if (i & (i + 1) % 16 == 0) printf("\n%d:\t", (i + 1) / 16);
    }
    printf("\nground-truth:\n");
    printf("%d:\t", 0);
    for (int i = 0; i < cut_off; i++) {
      printf("%.2f, ", hc2[i]);
      if (i & (i + 1) % 16 == 0) printf("\n%d:\t", (i + 1) / 16);
    }
  }
#endif
}

template <typename Element, typename ElementAcc>
__device__ void init_values(Element* a, Element* b, ElementAcc* c, int M, int N,
                            int K) {
  if (!thread0()) return;

  for (int i = 0; i < M * K; ++i) a[i] = static_cast<Element>(i % 2048 / 1000);

  for (int i = 0; i < K * N; ++i) b[i] = static_cast<Element>(i % 2048 / 1000);

  for (int i = 0; i < M * N; ++i) c[i] = 0.;
}

template <typename Element, typename ElementAcc,  //
          const int M, const int N, const int K,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename SharedC, typename RegC, typename StoreRegC>
__global__ void test_wmma(LoadRegA& load_rA, LoadRegB& load_rB,
                          StoreRegC& store_rC) {
  extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
  auto* shared_a = reinterpret_cast<Element*>(buf_);
  auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
  auto* shared_c =
      reinterpret_cast<ElementAcc*>(shared_b + TileIteratorB::Tile::kNumel);
  auto shared_ref =
      shared_c + TileIteratorA::Tile::kRows * TileIteratorB::Tile::kCols;

  init_values<Element, ElementAcc>(shared_a, shared_b, shared_c, M, N, K);

  __syncthreads();

  SharedC sC(shared_c);

  TileIteratorA sAs(shared_a);
  TileIteratorB sBs(shared_b);

  RegA rA;
  RegB rB;
  RegC acc;

  for (int k = 0; k < TileIteratorA::sc1; ++k) {
    auto sA = sAs(k);
    auto sB = sBs(k);

    load_rA(sA, rA);
    load_rB(sB, rB);
    __syncthreads();

    gemm(rA, rB, acc);
  }

  __syncthreads();

  store_rC(acc, sC);
  __syncthreads();

  if (thread0()) {
    __half* dA = reinterpret_cast<__half*>(shared_a);
    __half* dB = reinterpret_cast<__half*>(shared_b);
    float* dC = reinterpret_cast<float*>(shared_ref);
    naive_gemm(M, N, K, dA, dB, dC);

    check_results(dC, shared_c, M * N);
  }
}

}  // namespace

template <typename Element, typename ElementAcc, const int kM, const int kN,
          const int kK, typename WarpLayout>
struct TestTraits {
  static const int kThreads = tl::get_numel<WarpLayout> * 32;

  static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
  static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

  // ============= shared to register loader =================
  using MmaAtom = MmaAtom<Element, Element, ElementAcc, MMA_ATOM_16x16x16>;
  using BaseShape = MmaAtom::BaseTile;

  static constexpr int kAMs = kM / kWarpPerRow / BaseShape::kRows;
  static constexpr int kAKs = kK / BaseShape::kCols;

  static constexpr int kBKs = kK / BaseShape::kRows;
  static constexpr int kBNs = kN / kWarpPerCol / BaseShape::kCols;

  static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kRows;
  static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kCols;

  using SharedA = SharedTile<Element, tl::RowMajor<kM, kK>>;
  using TileIteratorA = STileIterator<SharedA, TileShape<kM, kK>>;

  using RegA = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<kAMs, kAKs>>;
  using LoadRegA =
      SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

  using SharedB = SharedTile<Element, tl::ColMajor<kK, kN>>;

  using RegB = RegTile<BaseTileColMajor<Element>, tl::ColMajor<kBKs, kBNs>>;
  using TileIteratorB = STileIterator<SharedB, TileShape<kK, kN>>;
  using LoadRegB =
      SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

  static_assert(TileIteratorA::sc1 == TileIteratorB::sc0,
                "dimension mismatch!");

  // ============= register to shared storer =================
  using SharedC = SharedTile<ElementAcc, tl::RowMajor<kM, kN>>;
  using RegC = RegTile<BaseTileRowMajor<ElementAcc>, tl::RowMajor<kCMs, kCNs>>;
  using StoreRegC = RegToSharedStorer<RegC, WarpLayout>;
};

template <const int kM, const int kN, const int kK, typename WarpLayout>
void run_test() {
  using Element = __half;
  using ElementAcc = float;

  using config = TestTraits<Element, ElementAcc, kM, kN, kK, WarpLayout>;

  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(config::kThreads, 1, 1);

  typename config::LoadRegA load_rA;
  typename config::LoadRegB load_rB;
  typename config::StoreRegC store_rC;

  using RegA = typename config::RegA;
  using RegB = typename config::RegB;
  using RegC = typename config::RegC;

  LOG(INFO) << std::endl
            << "RegA: " << RegA{} << std::endl
            << "RegB: " << RegB{} << std::endl
            << "RegC: " << RegC{} << std::endl;

  auto kernel =
      test_wmma<Element, ElementAcc, kM, kN, kK, typename config::TileIteratorA,
                RegA, typename config::LoadRegA, typename config::TileIteratorB,
                RegB, typename config::LoadRegB, typename config::SharedC, RegC,
                typename config::StoreRegC>;

  int shm_size =
      (kM + kN) * kK * sizeof(Element) + 2 * kM * kN * sizeof(ElementAcc);

  if (shm_size > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);
  }

  kernel<<<dim_grid, dim_block, shm_size>>>(load_rA, load_rB, store_rC);

  cudaDeviceSynchronize();

  LOG(INFO) << "[" << kM << ", " << kN << ", " << kK << "]. Test passed!"
            << std::endl;
}

TEST(TestWmma, test_m16n16k16_f) {
  run_test<16, 16, 64, tl::RowMajor<1, 1>>();
  run_test<32, 32, 64, tl::RowMajor<1, 1>>();
  run_test<64, 64, 64, tl::RowMajor<1, 1>>();

  run_test<32, 64, 64, tl::RowMajor<2, 1>>();
  run_test<32, 64, 64, tl::RowMajor<2, 1>>();
  run_test<64, 64, 64, tl::RowMajor<2, 1>>();
  run_test<128, 64, 64, tl::RowMajor<2, 1>>();
}

}  // namespace tilefusion::testing
