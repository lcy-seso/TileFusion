// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

#include <string>

namespace tilefusion {
// Returns the number of GPUs.
int GetDeviceCount();

// Returns the compute capability of the given GPU.
int GetComputeCapability(int id);

// Returns the number of multiprocessors for the given GPU.
int GetMultiProcessors(int id);

// Returns the maximum number of threads per multiprocessor for the given
// GPU.
int GetMaxThreadsPerMultiProcessor(int id);

// Returns the maximum number of threads per block for the given GPU.
int GetMaxThreadsPerBlock(int id);

// Returns the maximum shared memory per block for the given GPU.
int GetMaxSharedMemPerBlock(int id);

// Returns the maximum shared memory per multiprocessor for the given GPU.
int GetMaxSharedMemPerSM(int id);

// Returns the maximum grid size for the given GPU.
dim3 GetMaxGridDimSize(int id);

// Returns the name of the device.
std::string GetDeviceName();

}  // namespace tilefusion
