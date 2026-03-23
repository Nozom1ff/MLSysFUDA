#!/usr/bin/env python3
"""Test script to verify kernel compilation."""
import torch
import sys
sys.path.insert(0, '/home/nozom1/code/cuda/MLSysFUDA/solution/triton')

from kernel import kernel

def test_kernel_compile():
    """Test that the kernel compiles and runs without errors."""
    # Test parameters
    num_tokens = 2
    num_heads = 4
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 16
    num_pages = 8
    topk = 32
    sm_scale = 0.5

    device = torch.device('cuda')

    # Create test tensors
    q_nope = torch.randn(num_tokens, num_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Create sparse indices (some valid, some padding)
    sparse_indices = torch.randint(0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device=device)
    # Add some padding entries (-1)
    sparse_indices[0, -5:] = -1

    # Pre-allocate output
    output = torch.empty(num_tokens, num_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    # Run kernel
    kernel(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=sm_scale,
        output=output,
        lse=lse,
    )

    # Verify output shapes
    assert output.shape == (num_tokens, num_heads, head_dim_ckv), f"Output shape mismatch: {output.shape}"
    assert lse.shape == (num_tokens, num_heads), f"LSE shape mismatch: {lse.shape}"

    # Check that output is not all zeros (kernel did some computation)
    assert not torch.all(output == 0), "Output is all zeros!"

    print("✓ Kernel compilation and basic execution test PASSED")
    print(f"  - num_tokens: {num_tokens}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - head_dim_ckv: {head_dim_ckv}")
    print(f"  - head_dim_kpe: {head_dim_kpe}")
    print(f"  - topk: {topk}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - LSE shape: {lse.shape}")
    print(f"  - Output sample values: {output[0, 0, :5]}")

    # Test with different configurations
    print("\nTesting different configurations...")

    # Small head dim
    q_nope_small = torch.randn(1, 2, 64, dtype=torch.bfloat16, device=device)
    q_pe_small = torch.randn(1, 2, 32, dtype=torch.bfloat16, device=device)
    ckv_small = torch.randn(4, 8, 64, dtype=torch.bfloat16, device=device)
    kpe_small = torch.randn(4, 8, 32, dtype=torch.bfloat16, device=device)
    idx_small = torch.randint(0, 32, (1, 16), dtype=torch.int32, device=device)
    out_small = torch.empty(1, 2, 64, dtype=torch.bfloat16, device=device)
    lse_small = torch.empty(1, 2, dtype=torch.float32, device=device)

    kernel(q_nope_small, q_pe_small, ckv_small, kpe_small, idx_small, 0.125, out_small, lse_small)
    print("✓ Small head_dim (64) test PASSED")

    # Large TOPK
    q_nope_large = torch.randn(1, 2, 256, dtype=torch.bfloat16, device=device)
    q_pe_large = torch.randn(1, 2, 64, dtype=torch.bfloat16, device=device)
    ckv_large = torch.randn(64, 16, 256, dtype=torch.bfloat16, device=device)
    kpe_large = torch.randn(64, 16, 64, dtype=torch.bfloat16, device=device)
    idx_large = torch.randint(0, 1024, (1, 512), dtype=torch.int32, device=device)
    out_large = torch.empty(1, 2, 256, dtype=torch.bfloat16, device=device)
    lse_large = torch.empty(1, 2, dtype=torch.float32, device=device)

    kernel(q_nope_large, q_pe_large, ckv_large, kpe_large, idx_large, 0.125, out_large, lse_large)
    print("✓ Large TOPK (512) test PASSED")

    print("\n✓ All tests PASSED!")


if __name__ == "__main__":
    test_kernel_compile()
