"""Flash attention implementation for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "FlashAttention",
]


class FlashAttention:
    """A class implementing flash attention."""

    def __init__(
        self,
        softmax_scale: float,
        causal: bool,
    ) -> None:
        """Initialize the flash attention.

        Args:
            softmax_scale: Softmax scale.
            The scaling of QK^T before applying softmax.
                Default is 1.0 / sqrt(matrix_k).
            causal: bool. Whether to apply causal mask.
        """
        self.softmax_scale = softmax_scale
        self.causal = causal

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Perform the forward pass of flash attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            device: Device to run on.

        Returns:
            torch.Tensor: The attention output.
        """
        self.m, self.k = query.size(-2), query.size(-1)
        self.n, self.p = value.size(-2), value.size(-1)

        key = key.t()
        value = value.t()

        output = torch.empty(self.m, self.p, dtype=query.dtype, device=device)

        torch.ops.tilefusion.flash_attention(
            query,
            key,
            value,
            output,
            self.softmax_scale,
            self.causal,
        )

        return output
