// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/ops.hpp"

namespace tilefusion::kernels {

using namespace tilefusion;
namespace tl = tile_layout;

std::string generate_fused_two_gemms_kernel_source(const std::string& in_type,
                                                   const std::string& acc_type,
                                                   int m, int n, int k, int p) {
    std::stringstream ss;
    ss << R"(
#include "kernels/fused_two_gemms_device.cuh"

extern "C" __global__ void fused_two_gemms_kernel_)"
       << in_type << "_" << acc_type << "_" << m << "_" << n << "_" << k << "_"
       << p << R"((
    const )"
       << in_type << R"(* A,
    const )"
       << in_type << R"(* B,
    const )"
       << in_type << R"(* C,
    )" << in_type
       << R"(* D,
    int m, int n, int k, int p) {
    using Config = FusedTwoGemmsTraits<)"
       << in_type << ", " << acc_type << R"(,
        tl::RowMajor<2, 1>, )"
       << m << ", " << n << ", " << k << ", " << p << R"(>;
})";
    return ss.str();
}

void fused_two_gemms(const torch::Tensor& A, const torch::Tensor& B,
                     const torch::Tensor& C, torch::Tensor& D) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    CHECK_INPUT(D);

    const at::ScalarType dtype = A.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && B.scalar_type() == dtype &&
                    C.scalar_type() == dtype && D.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t m = A.size(0);
    const int64_t n = B.size(1);
    const int64_t k = B.size(1);
    const int64_t p = C.size(1);

    using WarpLayout = tl::RowMajor<2, 1>;
    using InType = __half;
    using AccType = float;

    std::string in_type = jit::get_type_string<InType>();
    std::string acc_type = jit::get_type_string<AccType>();

    // Generate kernel source
    std::string kernel_source =
        generate_fused_two_gemms_kernel_source(in_type, acc_type, m, n, k, p);

    // Create unique kernel name
    std::string kernel_name = "fused_two_gemms_kernel_" + in_type + "_" +
                              acc_type + "_" + std::to_string(m) + "_" +
                              std::to_string(n) + "_" + std::to_string(k) +
                              "_" + std::to_string(p);

    auto& jit = jit::JitCompiler::instance();
    CUfunction kernel = jit.get_or_compile_kernel(kernel_name, kernel_source);

    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    int block_size = 128;  // Adjust based on your needs
    int grid_size = (m * p + block_size - 1) / block_size;

    // Properly cast the tensor data pointers to half precision
    const InType* A_ptr =
        reinterpret_cast<const InType*>(A.data_ptr<at::Half>());
    const InType* B_ptr =
        reinterpret_cast<const InType*>(B.data_ptr<at::Half>());
    const InType* C_ptr =
        reinterpret_cast<const InType*>(C.data_ptr<at::Half>());
    InType* D_ptr = reinterpret_cast<InType*>(D.data_ptr<at::Half>());

    void* args[] = {(void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&D_ptr,
                    (void*)&m,     (void*)&n,     (void*)&k,     (void*)&p};

    CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1,
                                     0, nullptr, args, nullptr));

    LOG(INFO) << "Fused two gemms kernel launched successfully";
}
}  // namespace tilefusion::kernels
