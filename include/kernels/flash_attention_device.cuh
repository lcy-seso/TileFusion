// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tilefusion;
using namespace cell;
using namespace copy;
using namespace compute;
namespace tl = tile_layout;

namespace tilefusion::kernels {

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape>
struct FlashAttentionTraits {
    static constexpr int kSharedAccess = 64;

    using BaseShape = traits::BaseTileShape<InType>;

    using WarpLayout = tl::RowMajor<4, 1>;

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

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    // chunk the K dimension to fit into shared memory
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    using SharedA =
        SharedTile<InType, tl::RowMajor<kTM, kTK>, true, kSharedAccess>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAKs = kTK / BaseShape::kCols;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB =
        SharedTile<InType, tl::ColMajor<kTK, kTN>, true, kSharedAccess>;

    static constexpr int kBKs = kTK / BaseShape::kRows;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kCols;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using SharedBLoader = GlobalToSharedLoader<SharedB, WarpLayout>;
    using RegBLoader =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // operand C
    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
    // chunk the N dimension to fit into shared memory
    using GIteratorC = GTileIterator<GlobalC, TileShape<kTN, kTP>>;
    using SharedC =
        SharedTile<InType, tl::ColMajor<kTN, kTP>, true, kSharedAccess>;

    static constexpr int kCNs = kTN / BaseShape::kRows;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<OutType, tl::RowMajor<kTM, kTP>>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using RegDCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = RegToGlobalStorer<GlobalD, RegDCast, WarpLayout>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kCols;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    using RegAccPrinter = RegTilePrinter<RegAccCast, tl::Layout::kRowMajor>;

    // Convert the accumulator to half
    using ConvertHalf = RegTileConvert<RegAcc, RegAccCast>;
    using ConvertO = RegTileConvert<RegD, RegDCast>;

    using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;
    using RegVecPrinter = RegVecPrinter<RegVec>;

    using CopyVec = BaseTileCopy<RegVec>;
    using RowMax = MaxReduce<RegAccCast, tl::Layout::kRowMajor>;

    using RowSum = SumReduce<RegAccCast, tl::Layout::kRowMajor>;

    using Sub = BroadcastSub<RegVec, RegAccCast, tl::Layout::kRowMajor>;
    using Mul = BroadcastMul<RegVec, RegDCast, tl::Layout::kRowMajor>;
    using Div = BroadcastDiv<RegVec, RegDCast, tl::Layout::kRowMajor>;

    using Exp = RegTileExp<RegAccCast>;
    using Add = RegTileAdd<RegDCast>;

    using VecMax = BaseTileMax<RegVec>;
    using VecAdd = BaseTileAdd<RegVec>;
    using VecSub = BaseTileSub<RegVec>;
    using VecMul = BaseTileMul<RegVec>;
    using VecExp = BaseTileExp<RegVec>;

    using ApplyMask =
        ApplyMask<RegAcc, WarpLayout, BaseShape, MaskMode::kCausal>;
    using ApplyScoreScale = BroadcastScalarMul<RegAcc>;
};

template <typename InType, typename AccType, typename OutType,
          typename KeTraits>
__global__ void ke_flash_attention(const InType* dQ, const InType* dK,
                                   const InType* dV, OutType* dO, int kM,
                                   int kN, int kK, int kP, int kTM, int kTN,
                                   int kTK, int kTP, float softmax_scale,
                                   bool causal) {
    // Advance to the global data tile to the current CTA.
    const InType* Q = dQ + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* K = dK + blockIdx.z * (kK * kN);
    const InType* gV_ptr =
        dV + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    OutType* gO_ptr = dO + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                      (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sQ_ptr = shm;
    InType* sK_ptr = shm + KeTraits::SharedQ::kNumel;
    InType* sV_ptr =
        shm + KeTraits::SharedQ::kNumel + KeTraits::SharedK::kNumel;

    typename KeTraits::GIteratorQ gQs(Q);
    typename KeTraits::SharedQ sQ(sQ_ptr);
    typename KeTraits::RegQ rQ;

    typename KeTraits::SharedQLoader load_sq;
    typename KeTraits::RegQLoader load_rq;

    typename KeTraits::GIteratorK gKs(K);
    typename KeTraits::SharedK sK(sK_ptr);
    typename KeTraits::RegK rK;

    typename KeTraits::SharedKLoader load_sk;
    typename KeTraits::RegKLoader load_rk;

    typename KeTraits::GIteratorV gVs(gV_ptr);
    typename KeTraits::SharedV sV(sV_ptr);

    typename KeTraits::SharedVLoader load_sv;
    typename KeTraits::RegVLoader load_rv;
    typename KeTraits::RegV rV;

    typename KeTraits::RegO exp_values_f32;

    typename KeTraits::RegOCast rO;
    typename KeTraits::RegOCast exp_values;

    typename KeTraits::RegAcc attn_block_f32;
    typename KeTraits::RegAccCast attn_block;

    typename KeTraits::RegVec prev_norm_vec;
    typename KeTraits::RegVec cur_norm_vec;

    typename KeTraits::RegVec prev_max_vec;
    typename KeTraits::RegVec cur_max_vec;
    typename KeTraits::RegVec new_max_vec;

    typename KeTraits::RegVec prev_sum_vec;
    typename KeTraits::RegVec cur_sum_vec;
    typename KeTraits::RegVec new_sum_vec;

    typename KeTraits::RegVec prev_norm_mul_sum;
    typename KeTraits::RegVec cur_norm_mul_sum;
    typename KeTraits::RegVec prev_sum_mul_norm;

    typename KeTraits::RowMax row_max;
    typename KeTraits::RowSum row_sum;
    typename KeTraits::CopyVec copy_vec;

    typename KeTraits::ConvertAcc cast_acc;  // Convert acc to half precision
    typename KeTraits::ConvertO cast_o;      // Convert half precision to float.

    typename KeTraits::Sub sub;
    typename KeTraits::Mul mul;
    typename KeTraits::Div div;

    typename KeTraits::Exp exp;
    typename KeTraits::Add add;

    typename KeTraits::VecMax vec_max;
    typename KeTraits::VecAdd vec_add;
    typename KeTraits::VecSub vec_sub;
    typename KeTraits::VecMul vec_mul;
    typename KeTraits::VecExp vec_exp;

    typename KeTraits::ApplyMask apply_mask;
    typename KeTraits::ApplyScoreScale apply_score_scale;

    for (int n = 0; n < KeTraits::GIteratorV::sc0; ++n) {
        load_sv(gVs(n), sV);

        for (int k = 0; k < KeTraits::GIteratorQ::sc1; ++k) {
            load_sq(gQs(k), sQ);
            load_sk(gKs(k, n), sK);
            __copy_async();
            __syncthreads();

            load_rq(sQ, rQ);
            load_rk(sK, rK);
            __syncthreads();

            gemm(rQ, rK, attn_block_f32);
        }
        load_rv(sV, rV);
        __syncthreads();

        if (causal) {
            apply_mask(attn_block_f32, blockIdx.x * kTM, n * kTN, -INFINITY);
        }

        apply_score_scale(attn_block_f32, softmax_scale, attn_block_f32);

        cast_acc(attn_block_f32, attn_block);

        // Compute row max.
        row_max(attn_block, cur_max_vec);

        // Broadcast subtract from `attn_block`.
        sub(cur_max_vec, attn_block);

        // Compute exp in `attn_block`.
        exp(attn_block, attn_block);

        // Compute `cur_sum_vec` by reduce sum of `attn_block`.
        row_sum(attn_block, cur_sum_vec);

        // Compute new max vector.
        vec_max(cur_max_vec, prev_max_vec, new_max_vec);

        // Renormalization for the previous block.
        vec_sub(prev_max_vec, new_max_vec, prev_norm_vec);
        vec_exp(prev_norm_vec, prev_norm_vec);

        // Renormalization for the current block.
        vec_sub(cur_max_vec, new_max_vec, cur_norm_vec);
        vec_exp(cur_norm_vec, cur_norm_vec);

        // Update normalization factor l(x)
        vec_mul(prev_norm_vec, prev_sum_vec, prev_norm_mul_sum);
        vec_mul(cur_norm_vec, cur_sum_vec, cur_norm_mul_sum);
        vec_add(prev_norm_mul_sum, cur_norm_mul_sum, new_sum_vec);

        // Compute unnormized attention block.
        gemm(attn_block, rV, exp_values_f32);

        cast_o(exp_values_f32, exp_values);

        mul(prev_norm_mul_sum, rO);

        mul(cur_norm_vec, exp_values);

        add(rO, exp_values, rO);

        // Normalize the attention block.
        div(new_sum_vec, rO);

        // Update max vector and sum vector.
        copy_vec(new_max_vec, prev_max_vec);
        copy_vec(new_sum_vec, prev_sum_vec);

        // Clear the accumulator.
        attn_block_f32.clear();
        exp_values_f32.clear();
    }
    __syncthreads();

    // Store O tile from register to global.
    typename KeTraits::GlobalO gO(gO_ptr);
    typename KeTraits::OStorer storer_o;
    storer_o(rO, gO);
}

}  // namespace tilefusion::kernels
