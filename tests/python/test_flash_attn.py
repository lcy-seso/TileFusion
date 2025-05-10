"""Test flash attention implementation.

isort:skip_file
"""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any

import pytest
import torch
import math

from tilefusion.ops import FlashAttention

# import pdb


class FlashAttentionRef:
    """Reference implementation of flash attention."""

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        tile_p: int,
        softmax_scale: float,
        causal: bool,
    ) -> None:
        """Initialize the flash attention.

        Args:
            tile_m: Tile size for M dimension.
            tile_n: Tile size for N dimension.
            tile_k: Tile size for K dimension.
            tile_p: Tile size for P dimension.
            softmax_scale: Softmax scale.
                The scaling of QK^T before applying softmax.
                Default is 1.0 / sqrt(matrix_k).
            causal: bool. Whether to apply causal mask.
        """
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.tile_p = tile_p

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
        length_qk = query.shape[0]
        length_v = value.shape[0]
        hidden_v = value.shape[1]

        output = torch.empty(length_qk, hidden_v, device=device)
        iter_n = length_v // self.tile_n

        prev_maxes = torch.zeros(length_qk, 1, device=device)
        prev_sums = torch.zeros(length_qk, 1, device=device)

        ks = torch.chunk(key, iter_n, dim=-1)
        vs = torch.chunk(value, iter_n, dim=-2)

        for chunk_idx in range(iter_n):
            key_chunk = ks[chunk_idx]
            value_chunk = vs[chunk_idx]

            attn_weights = query @ key_chunk  # m * ktn

            if self.causal:
                # Create a causal mask for the attention weights.
                mask = torch.tril(
                    torch.ones_like(attn_weights), diagonal=0
                ).bool()
                attn_weights = torch.where(mask, attn_weights, float("-inf"))

            attn_weights = attn_weights * self.softmax_scale

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            exp_weights = torch.exp(attn_weights - cur_maxes)
            # unnormalized attention score @ values
            exp_values = exp_weights @ value_chunk
            # move the normalization step to the very end of the attention
            # computation.
            cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True)  # l(x_cur)

            # =========   renormalization  ======================#
            new_maxes = torch.max(cur_maxes, prev_maxes)  # update m(x)
            # renormalization factor for the previous block
            renorm_prev = torch.exp(prev_maxes - new_maxes)
            # renormalization factor for the current block
            renorm_cur = torch.exp(cur_maxes - new_maxes)

            # update normalization factor l(x)
            new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

            output = output * prev_sums * renorm_prev + renorm_cur * exp_values
            output /= new_sums

            prev_sums = new_sums
            prev_maxes = new_maxes

        return output


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


def run_flash_attention(
    length_qk: int,
    length_v: int,
    hidden_qk: int,
    hidden_v: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    tile_p: int,
    softmax_scale: float,
    causal: bool,
    eps: float = 8e-2,
) -> None:
    """Run flash attention test with given dimensions.

    Args:
        length_qk: Length of the query sequence.
        length_v: Length of the key/value sequence.
        hidden_qk: Hidden dimension of the query and key.
        hidden_v: Hidden dimension of the value.
        tile_m: Tile size for M dimension.
        tile_n: Tile size for N dimension.
        tile_k: Tile size for K dimension.
        tile_p: Tile size for P dimension.
        softmax_scale: Softmax scale.
            The scaling of QK^T before applying softmax.
            Default is 1.0 / sqrt(matrix_k).
        causal: bool. Whether to apply causal mask.
        eps: float. The epsilon value for the test.
    """
    query = torch.randn(
        length_qk, hidden_qk, dtype=torch.float16, device="cuda"
    )
    key = torch.randn(length_qk, hidden_v, dtype=torch.float16, device="cuda")
    value = torch.randn(length_v, hidden_v, dtype=torch.float16, device="cuda")

    flash_attn = FlashAttentionRef(
        tile_m,
        tile_n,
        tile_k,
        tile_p,
        softmax_scale,
        causal,
    )
    ref_output = flash_attn(query, key, value)

    flash_attention = FlashAttention(
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = flash_attention(query, key, value)
    assert torch.allclose(output, ref_output, atol=eps)


# @pytest.mark.parametrize(
#     "test_case",
#     [
#         {
#             "name": "test_case1",
#             "matrix_m": 64,
#             "matrix_n": 128,
#             "matrix_k": 128,
#             "matrix_p": 128,
#             "tile_m": 64,
#             "tile_n": 128,
#             "tile_k": 128,
#             "tile_p": 128,
#             "softmax_scale": 1.0 / math.sqrt(128),
#             "causal": False,
#         },
#         {
#             "name": "test_case2",
#             "matrix_m": 64,
#             "matrix_n": 256,
#             "matrix_k": 128,
#             "matrix_p": 128,
#             "tile_m": 64,
#             "tile_n": 128,
#             "tile_k": 128,
#             "tile_p": 128,
#             "softmax_scale": 1.0 / math.sqrt(128),
#             "causal": False,
#         },
#         {
#             "name": "test_case3",
#             "matrix_m": 128,
#             "matrix_n": 128,
#             "matrix_k": 128,
#             "matrix_p": 128,
#             "tile_m": 64,
#             "tile_n": 128,
#             "tile_k": 128,
#             "tile_p": 128,
#             "softmax_scale": 1.0 / math.sqrt(128),
#             "causal": True,
#         },
#     ],
#     ids=lambda x: x["name"],
# )
def test_flash_attention(test_case: dict[str, Any]) -> None:
    """Test flash attention with different matrix dimensions.

    Args:
        test_case: Dictionary containing test parameters
    """
    run_flash_attention(
        length_qk=test_case["length_qk"],
        length_v=test_case["length_v"],
        hidden_qk=test_case["hidden_qk"],
        hidden_v=test_case["hidden_v"],
        tile_m=test_case["tile_m"],
        tile_n=test_case["tile_n"],
        tile_k=test_case["tile_k"],
        tile_p=test_case["tile_p"],
        softmax_scale=test_case["softmax_scale"],
        causal=test_case["causal"],
    )


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])

    run_flash_attention(
        length_qk=64,
        length_v=128,
        hidden_qk=128,
        hidden_v=128,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        tile_p=128,
        softmax_scale=1.0 / math.sqrt(128),
        causal=False,
    )
