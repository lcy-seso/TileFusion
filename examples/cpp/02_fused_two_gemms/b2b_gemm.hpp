// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tilefusion;
using namespace tilefusion::cell;
using namespace tilefusion::cell::copy;
namespace tl = tile_layout;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, typename WarpLayout>
struct B2BGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static const bool kSwizzled = true;

    ///====== Tiles
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kSwizzled>;
    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kTK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kSwizzled>;
    static constexpr int kBKs = kTK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
    using SharedC = SharedTile<InType, tl::ColMajor<kTN, kTP>, kSwizzled>;
    static constexpr int kCNs = kTN / BaseShape::kTileSize;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using GlobalD = GlobalTile<AccType, tl::RowMajor<kTM, kTP>>;
    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;

    /// accumulator on register
    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegAcc =  // Reg Acc
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    ///======= Tile Iterators
    // chunk the K dimension to fit into shared memory
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;
    // chunk the N dimension to fit into shared memory
    using GIteratorC = GTileIterator<GlobalC, TileShape<kTN, kTP>>;

    ///====== Loader and Storer
    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    using SharedBLoader = GlobalToSharedLoader<SharedB, WarpLayout>;
    using RegBLoader =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    using DStorer = copy::RegToGlobalStorer<GlobalD, RegD, WarpLayout>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
};

template <typename InType, typename AccType,                     //
          typename GIteratorA, typename SharedA, typename RegA,  //
          typename SharedALoader, typename RegALoader,           //
          typename GIteratorB, typename SharedB, typename RegB,  //
          typename SharedBLoader, typename RegBLoader,           //
          typename GIteratorC, typename SharedC, typename RegC,  //
          typename SharedCLoader, typename RegCLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalD, typename RegD,
          typename DStorer, typename ConvertAcc>
__global__ void KeBack2BackGemm(const InType* dA, const InType* dB,
                                const InType* dC, AccType* dD, int kM, int kN,
                                int kK, int kP, int kTM, int kTN, int kTK,
                                int kTP) {
    // Advance to the global data tile to the current CTA.
    const InType* A = dA + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* B = dB + blockIdx.z * (kK * kN);
    const InType* gC_ptr =
        dC + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    AccType* gD_ptr = dD + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                      (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sA_ptr = shm;
    InType* sB_ptr = shm + SharedA::kNumel;
    InType* sC_ptr = shm + SharedA::kNumel + SharedB::kNumel;

    GIteratorA gAs(A);
    SharedA sA(sA_ptr);
    RegA rA;

    SharedALoader load_sa;
    RegALoader load_ra;

    GIteratorB gBs(B);
    SharedB sB(sB_ptr);
    RegB rB;

    SharedBLoader load_sb;
    RegBLoader load_rb;

    GIteratorC gCs(gC_ptr);
    SharedC sC(sC_ptr);

    SharedCLoader load_sc;
    RegCLoader load_rc;
    RegC rC;

    RegD rD;
    RegAcc acc;
    RegAccCast acc_half;

    ConvertAcc cast_acc;

    GlobalD gD(gD_ptr);
    DStorer storer_d;

    for (int n = 0; n < GIteratorC::sc0; ++n) {
        load_sc(gCs(n), sC);

        for (int k = 0; k < GIteratorA::sc1; ++k) {
            load_sa(gAs(k), sA);
            load_sb(gBs(k, n), sB);
            __copy_async();
            __syncthreads();

            load_ra(sA, rA);
            load_rb(sB, rB);
            __syncthreads();

            gemm(rA, rB, acc);
        }
        load_rc(sC, rC);
        __syncthreads();

        cast_acc(acc, acc_half);

        gemm(acc_half, rC, rD);
        acc.clear();
    }
    __syncthreads();

    storer_d(rD, gD);
}
