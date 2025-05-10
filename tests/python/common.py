"""Common utilities for testing."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = [
    "random_tensor",
]

import torch


def random_tensor(
    size: tuple[int, ...],
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a tensor with the given parameters."""
    return torch.randn(size, device=device, dtype=dtype)
