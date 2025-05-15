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

// kernel wrapper
template <typename InType, typename AccType, typename OutType, typename Config>
__attribute__((global)) void kernel_wrapper(const InType* A, const InType* B,
                                            const InType* C, OutType* D) {
    ke_flash_attention<InType, AccType, OutType, Config>(A, B, C, D);
}

void flash_attention(const torch::Tensor& Q, const torch::Tensor& K,
                     const torch::Tensor& V, torch::Tensor& O,  //
                     int64_t tile_length_q, int64_t tile_length_kv,
                     int64_t tile_hidden_qk, int64_t tile_hidden_v,
                     double softmax_scale, bool causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(O);

    const at::ScalarType dtype = Q.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && K.scalar_type() == dtype &&
                    V.scalar_type() == dtype && O.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t batch_size = Q.size(0);
    const int64_t length_q = Q.size(1);
    const int64_t length_kv = K.size(2);
    const int64_t hidden_qk = Q.size(2);
    const int64_t hidden_v = V.size(1);

    using InType = __half;
    using AccType = float;
    using OutType = __half;

    using WholeShape = TileShape<256, 256, 128, 128>;
    using CtaTileShape = TileShape<128, 128, 128, 128>;

    using Config = FlashAttentionTraits<InType, AccType, OutType, WholeShape,
                                        CtaTileShape, 1.0, false>;

    int block_x = ceil_div(length_q, tile_length_q);
    int block_y = ceil_div(hidden_v, tile_hidden_v);

    dim3 grid(block_x, block_y, batch_size);
    dim3 block(Config::kThreads, 1, 1);

    std::cout << std::endl
              << "RegQ:" << typename Config::RegQ{} << std::endl
              << "RegK: " << typename Config::RegK{} << std::endl
              << std::endl;

    std::cout << "batch_size: " << batch_size << std::endl
              << "length_q: " << length_q << std::endl
              << "length_kv: " << length_kv << std::endl
              << "hidden_qk: " << hidden_qk << std::endl
              << "hidden_v: " << hidden_v << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")"
              << std::endl;
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z
              << ")" << std::endl;

    int shm_input =
        (tile_length_q * tile_hidden_qk + tile_hidden_qk * tile_length_kv +
         tile_length_kv * tile_hidden_v);
    int shm_output = tile_length_q * tile_hidden_v;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(OutType)
                                          : shm_input * sizeof(InType);

    auto kernel = &kernel_wrapper<InType, AccType, OutType, Config>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    auto dQ = reinterpret_cast<const InType*>(Q.data_ptr());
    auto dK = reinterpret_cast<const InType*>(K.data_ptr());
    auto dV = reinterpret_cast<const InType*>(V.data_ptr());
    auto dO = reinterpret_cast<OutType*>(O.data_ptr());

    kernel<<<grid, block, shm_size, 0>>>(dQ, dK, dV, dO);

    cudaDeviceSynchronize();
}

}  // namespace tilefusion::kernels
