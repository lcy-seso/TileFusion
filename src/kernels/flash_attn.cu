// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/flash_attention_device.cuh"
#include "kernels/ops.hpp"
#include "types/mod.hpp"

namespace tilefusion::kernels {

template <const int kM, const int kN, const int kK, const int kP>
using FlashAttentionShape = TileShape<kM, kN, kK, kP>;

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape, const int kBatch>
void run(const InType* dQ, const InType* dK, const InType* dV, OutType* dO,
         float softmax_scale, bool causal) {
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    using Config = FlashAttentionTraits<InType, AccType, OutType, WholeShape,
                                        CtaTileShape>;

    // using RegA = typename Config::RegA;
    // using RegB = typename Config::RegB;
    // using RegC = typename Config::RegC;
    // using RegD = typename Config::RegD;
    // using RegDCast = typename Config::RegDCast;
    // using RegAcc = typename Config::RegAcc;
    // using RegAccCast = typename Config::RegAccCast;

    // using RegAccPrinter = typename Config::RegAccPrinter;

    // using GIteratorA = typename Config::GIteratorA;
    // using SharedA = typename Config::SharedA;
    // using SharedALoader = typename Config::SharedALoader;
    // using RegALoader = typename Config::RegALoader;

    // using GIteratorB = typename Config::GIteratorB;
    // using SharedB = typename Config::SharedB;
    // using SharedBLoader = typename Config::SharedBLoader;
    // using RegBLoader = typename Config::RegBLoader;

    // using GIteratorC = typename Config::GIteratorC;
    // using SharedC = typename Config::SharedC;
    // using SharedCLoader = typename Config::SharedCLoader;
    // using RegCLoader = typename Config::RegCLoader;

    // using DStorer = typename Config::DStorer;

    // using ConvertAcc = typename Config::ConvertHalf;
    // using ConvertO = typename Config::ConvertO;

    // using RegVec = typename Config::RegVec;
    // using RegVecPrinter = typename Config::RegVecPrinter;

    // using CopyVec = typename Config::CopyVec;
    // using RowMax = typename Config::RowMax;
    // using RowSum = typename Config::RowSum;

    // using BroadcastSub = typename Config::BroadcastSub;
    // using BroadcastMul = typename Config::BroadcastMul;
    // using BroadcastDiv = typename Config::BroadcastDiv;

    // using BlockExp = typename Config::BlockExp;
    // using BlockAdd = typename Config::BlockAdd;

    // using VecMax = typename Config::VecMax;
    // using VecAdd = typename Config::VecAdd;
    // using VecSub = typename Config::VecSub;
    // using VecMul = typename Config::VecMul;
    // using VecExp = typename Config::VecExp;

    // using ApplyScoreScale = typename Config::ApplyScoreScale;
    // using ApplyMask = typename Config::ApplyMask;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto kernel = &ke_flash_attention<InType, AccType, OutType, Config>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size, 0>>>(dQ, dK, dV, dO, kM, kN, kK, kP, kTM,
                                         kTN, kTK, kTP, softmax_scale, causal);

    cudaDeviceSynchronize();
}

void flash_attention(const torch::Tensor& Q, const torch::Tensor& K,
                     const torch::Tensor& V, torch::Tensor& O,
                     double softmax_scale, bool causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(O);

    const at::ScalarType dtype = Q.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && K.scalar_type() == dtype &&
                    V.scalar_type() == dtype && O.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t m = Q.size(0);
    const int64_t n = K.size(0);
    const int64_t k = K.size(1);
    const int64_t p = V.size(0);

    using InType = __half;
    using AccType = float;
    using OutType = __half;

    using CtaTileShape = TileShape<64, 128, 128, 128>;

    auto dQ = reinterpret_cast<const InType*>(Q.data_ptr());
    auto dK = reinterpret_cast<const InType*>(K.data_ptr());
    auto dV = reinterpret_cast<const InType*>(V.data_ptr());
    auto dO = reinterpret_cast<OutType*>(O.data_ptr());

    // if (m == 64 && n == 256 && k == 128 && p == 128) {
    //     using WholeShape = FlashAttentionShape<64, 256, 128, 128>;
    //     const int kBatch = 1;
    //     run<InType, AccType, OutType, WholeShape, CtaTileShape, kBatch>(
    //         dQ, dK, dV, dO, softmax_scale, causal);
    // } else if (m == 64 && n == 128 && k == 128 && p == 128) {
    //     using WholeShape = FlashAttentionShape<64, 128, 128, 128>;
    //     const int kBatch = 1;
    //     run<InType, AccType, OutType, WholeShape, CtaTileShape, kBatch>(
    //         dQ, dK, dV, dO, softmax_scale, causal);
    // } else if (m == 128 && n == 128 && k == 128 && p == 128) {
    //     using WholeShape = FlashAttentionShape<128, 128, 128, 128>;
    //     const int kBatch = 1;
    //     run<InType, AccType, OutType, WholeShape, CtaTileShape, kBatch>(
    //         dQ, dK, dV, dO, softmax_scale, causal);
    // } else {
    //     throw std::runtime_error("Unsupported shape");
    // }
}

}  // namespace tilefusion::kernels
