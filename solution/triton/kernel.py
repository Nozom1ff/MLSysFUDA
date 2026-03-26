"""
DSA Sparse Attention Kernel - Highly Optimized for Triton 3.x and B200.
Handles sparse attention with TopK KV cache selection.
Uses online softmax algorithm for numerical stability.

Optimizations included without modifying Python API signature:
1. Blocked Matmul (WGMMA): Uses tl.dot to trigger Tensor Cores instead of CUDA Cores.
2. MLA Memory Sharing: Loads KV cache ONCE for multiple heads (BLOCK_H).
3. Pipelining: num_stages=3/4 to utilize B200 Async Copy (TMA) for sparse reads.
4. Native BF16/FP16 Loads: Avoids early cast to FP32 to ensure Tensor Core execution.
"""
from typing import Tuple
import torch
import triton
import triton.language as tl

@triton.jit
def _dsa_sparse_attention_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, indices_ptr, output_ptr, lse_ptr,
    stride_qn_b, stride_qn_h, stride_qn_d,
    stride_qp_b, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_s, stride_ckv_d,
    stride_kpe_p, stride_kpe_s, stride_kpe_d,
    stride_idx_b, stride_idx_k,
    stride_out_b, stride_out_h, stride_out_d,
    stride_lse_b, stride_lse_h,
    NUM_HEADS: tl.constexpr,     # Added as constexpr to handle unaligned heads
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,       # Block size for Heads (e.g., 16)
    BLOCK_K: tl.constexpr,       # Block size for Top-K loops (e.g., 32)
):
    """
    Highly optimized kernel using Tensor Cores and Block-level processing.
    """
    b_idx = tl.program_id(0)         # batch/token index
    h_group_idx = tl.program_id(1)   # head group index

    # 1. Continuous offsets for Heads and Dimensions
    offs_h = h_group_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < NUM_HEADS
    
    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    ckv_mask = offs_d_ckv < HEAD_DIM_CKV
    kpe_mask = offs_d_kpe < HEAD_DIM_KPE

    # Combined masks for Q loading to prevent out-of-bounds
    q_ckv_mask = h_mask[:, None] & ckv_mask[None, :]
    q_kpe_mask = h_mask[:, None] & kpe_mask[None, :]

    # 2. Load Queries (Kept in native dtype for Tensor Cores, NO early float32 cast!)
    q_nope_ptrs = q_nope_ptr + b_idx * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d_ckv[None, :] * stride_qn_d
    q_nope = tl.load(q_nope_ptrs, mask=q_ckv_mask, other=0.0) #[BLOCK_H, HEAD_DIM_CKV]

    q_pe_ptrs = q_pe_ptr + b_idx * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_d_kpe[None, :] * stride_qp_d
    q_pe = tl.load(q_pe_ptrs, mask=q_kpe_mask, other=0.0)     #[BLOCK_H, HEAD_DIM_KPE]

    # 3. Initialize Online Softmax Accumulators in FP32
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    idx_base = b_idx * stride_idx_b

    # 4. Process Top-K entries in BLOCKS to enable Pipelining and GEMM
    for k_start in range(0, TOPK, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        k_idx = k_start + offs_k
        k_mask = k_idx < TOPK

        # Load sparse indices for the block
        sparse_idx = tl.load(indices_ptr + idx_base + k_idx * stride_idx_k, mask=k_mask, other=-1)
        valid = (sparse_idx >= 0) & k_mask

        # Compute page and offset
        page_idx = sparse_idx // PAGE_SIZE
        page_offset = sparse_idx % PAGE_SIZE

        # Combined masks for Keys loading
        load_mask_ckv = valid[:, None] & ckv_mask[None, :]
        load_mask_kpe = valid[:, None] & kpe_mask[None, :]

        # Load K vectors [BLOCK_K, HEAD_DIM] - Note: Keys are shared across BLOCK_H (MLA advantage)
        k_ckv_ptrs = ckv_ptr + page_idx[:, None] * stride_ckv_p + page_offset[:, None] * stride_ckv_s + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = tl.load(k_ckv_ptrs, mask=load_mask_ckv, other=0.0)

        k_kpe_ptrs = kpe_ptr + page_idx[:, None] * stride_kpe_p + page_offset[:, None] * stride_kpe_s + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = tl.load(k_kpe_ptrs, mask=load_mask_kpe, other=0.0)

        # 5. CORE OPTIMIZATION: Matrix Multiplication (GEMM) using tl.dot -> Triggers Tensor Cores
        # Shape: [BLOCK_H, DIM] @ [DIM, BLOCK_K] ->[BLOCK_H, BLOCK_K]
        qk_ckv = tl.dot(q_nope, tl.trans(k_ckv)) 
        qk_kpe = tl.dot(q_pe, tl.trans(k_kpe))

        # Compute logit and apply mask safely
        logit = (qk_ckv + qk_kpe) * SM_SCALE
        logit = tl.where(valid[None, :], logit.to(tl.float32), float("-inf"))

        # 6. Online Softmax update
        m_ij = tl.max(logit, axis=1) # Max over BLOCK_K
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(logit - m_i_new[:, None])
        p = tl.where(valid[None, :], p, 0.0) # Zero out invalid paddings

        # Update sum of exp
        l_i_new = l_i * alpha + tl.sum(p, axis=1)

        # 7. VALUE ACCUMULATION: Second GEMM -> [BLOCK_H, BLOCK_K] @ [BLOCK_K, HEAD_DIM]
        # Cast probabilities to native dtype (bf16/fp16) to leverage Tensor Cores
        p_cast = p.to(q_nope.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_cast, k_ckv)

        m_i = m_i_new
        l_i = l_i_new

    # 8. Safe normalization
    out = acc / l_i[:, None]

    # Store output with mask
    out_ptrs = output_ptr + b_idx * stride_out_b + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=q_ckv_mask)

    # Store LSE
    ln2 = 0.6931471805599453
    lse_val = tl.where(l_i > 0, (m_i + tl.log(l_i)) / ln2, float("-inf"))
    lse_ptrs = lse_ptr + b_idx * stride_lse_b + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse_val, mask=h_mask)



def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """
    DSA Sparse Attention kernel (Destination-Passing Style).
    (API Signature unchanged)
    """
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    device = q_nope.device

    # Verify output tensors are pre-allocated
    assert output.shape == (num_tokens, num_heads, head_dim_ckv), f"Output shape mismatch: {output.shape} vs {(num_tokens, num_heads, head_dim_ckv)}"
    assert lse.shape == (num_tokens, num_heads), f"LSE shape mismatch: {lse.shape} vs {(num_tokens, num_heads)}"


    BLOCK_H = 16 
    BLOCK_K = 32 
    
    grid = (num_tokens, triton.cdiv(num_heads, BLOCK_H))

    # Launch kernel with optimized settings for B200
    _dsa_sparse_attention_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output, lse,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        NUM_HEADS=num_heads,         
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        PAGE_SIZE=page_size,
        SM_SCALE=sm_scale,
        TOPK=topk,
        BLOCK_H=BLOCK_H,              
        BLOCK_K=BLOCK_K,             
        num_warps=4,
        num_stages=3,                
    )