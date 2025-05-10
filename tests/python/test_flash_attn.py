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

# import pdb

import tilefusion
from tests.python.common import random_tensor


class AttentionRef:
    """Reference implementation of multi-head attention."""

    def __init__(
        self,
        length_q: int,
        length_kv: int,
        softmax_scale: float,
        is_causal: bool = False,
    ) -> None:
        """Initialize the multi-head attention.

        Args:
            length_q: Length of the query sequence.
            length_kv: Length of the key/value sequence.
            softmax_scale: Softmax scale.
            is_causal: Whether to apply causal mask.
        """
        self.is_causal = is_causal
        self.mask = torch.triu(torch.ones(length_q, length_kv), diagonal=1)
        self.softmax_scale = softmax_scale

    def __call__(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass of multi-head attention.

        Args:
            queries: Query tensor, shape (batch_size, length_qk, hidden_qk).
            keys: Key tensor, shape (batch_size, length_qk, hidden_v).
            values: Value tensor, shape (batch_size, length_v, hidden_v).

        Returns:
            torch.Tensor: The attention output.
        """
        attn_scores = queries @ keys.transpose(1, 2)
        if self.is_causal:
            attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores * self.softmax_scale, dim=-1)
        return attn_weights @ values


def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Compute multi-head attention using a reference implementation.

    This is a helper function that creates an AttentionRef instance and applies
    it to compute attention scores. Used as a reference implementation to
    validate the flash attention implementation.

    Args:
        query: Query tensor of shape (batch_size, length_q, hidden_qk)
        key: Key tensor of shape (batch_size, length_kv, hidden_qk)
        value: Value tensor of shape (batch_size, length_kv, hidden_v)
        softmax_scale: Scale factor applied before softmax
                       (typically 1/sqrt(hidden_qk))
        causal: Whether to apply causal masking to prevent attention to
        future tokens

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, length_q, hidden_v)
            containing the attention-weighted combination of values
    """
    length_q = query.shape[1]
    length_kv = key.shape[1]
    mha = AttentionRef(length_q, length_kv, softmax_scale, causal)
    return mha(query, key, value)


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


def run_flash_attention(
    batch_size: int,
    length_q: int,
    length_kv: int,
    hidden_qk: int,
    hidden_v: int,
    tile_length_q: int,
    tile_hidden_qk: int,
    tile_length_kv: int,
    tile_hidden_v: int,
    softmax_scale: float,
    causal: bool,
    eps: float = 8e-2,
) -> None:
    """Run flash attention test with given parameters.

    This function creates random tensors for query, key, and value,
    then compares the output of the flash attention implementation
    against a reference implementation.

    Args:
        batch_size: Number of sequences in the batch
        length_q: Length of query sequence
        length_kv: Length of key/value sequence
        hidden_qk: Hidden dimension size for query and key
        hidden_v: Hidden dimension size for value
        tile_length_q: Tile size for query sequence length
        tile_hidden_qk: Tile size for query/key hidden dimension
        tile_length_kv: Tile size for key/value sequence length
        tile_hidden_v: Tile size for value hidden dimension
        softmax_scale: Scale factor for attention scores before softmax
        causal: Whether to use causal attention masking
        eps: Tolerance for numerical comparison between implementations

    Returns:
        None. Raises AssertionError if outputs don't match within tolerance.
    """
    if length_q != length_kv:
        raise ValueError("length_q must be equal to length_kv")

    query = random_tensor((batch_size, length_q, hidden_qk))
    key = random_tensor((batch_size, length_kv, hidden_qk))
    value = random_tensor((batch_size, length_kv, hidden_v))

    ref_output = mha(query, key, value, softmax_scale, causal)

    print("ref_output")  # noqa: T201
    print(ref_output)  # noqa: T201

    output = tilefusion.ops.flash_attention(
        query,
        key,
        value,
        tile_length_q=tile_length_q,
        tile_length_kv=tile_length_kv,
        tile_hidden_qk=tile_hidden_qk,
        tile_hidden_v=tile_hidden_v,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    print("output")  # noqa: T201
    # print(output)  # noqa: T201

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
        batch_size=test_case["batch_size"],
        length_q=test_case["length_q"],
        length_kv=test_case["length_kv"],
        hidden_qk=test_case["hidden_qk"],
        hidden_v=test_case["hidden_v"],
        tile_length_q=test_case["tile_length_q"],
        tile_hidden_qk=test_case["tile_hidden_qk"],
        tile_length_kv=test_case["tile_length_kv"],
        tile_hidden_v=test_case["tile_hidden_v"],
        softmax_scale=test_case["softmax_scale"],
        causal=test_case["causal"],
    )


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])

    run_flash_attention(
        batch_size=1,
        length_q=256,
        length_kv=256,
        hidden_qk=64,
        hidden_v=64,
        tile_length_q=128,
        tile_hidden_qk=128,
        tile_length_kv=64,
        tile_hidden_v=64,
        softmax_scale=1.0 / math.sqrt(128),
        causal=False,
    )
