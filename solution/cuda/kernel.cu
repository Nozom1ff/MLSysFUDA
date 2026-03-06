"""
DSA Sparse Attention Kernel - Fixed for Triton 3.x and B200.
Handles sparse attention with TopK KV cache selection.
Uses online softmax algorithm for numerical stability.
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
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
):
    """
    Main kernel: compute sparse attention for one (token, head) pair.
    Uses online softmax algorithm for numerical stability.
    """
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

#Dimension offsets with mask for safety
    offs_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_kpe = tl.arange(0, HEAD_DIM_KPE)

#Create masks for dimension bounds
    ckv_mask = offs_ckv < HEAD_DIM_CKV
    kpe_mask = offs_kpe < HEAD_DIM_KPE

#Load query vectors with mask
    q_nope_ptrs = q_nope_ptr + b_idx * stride_qn_b + h_idx * stride_qn_h + offs_ckv * stride_qn_d
    q_nope = tl.load(q_nope_ptrs, mask=ckv_mask, other=0.0).to(tl.float32)
    
    q_pe_ptrs = q_pe_ptr + b_idx * stride_qp_b + h_idx * stride_qp_h + offs_kpe * stride_qp_d
    q_pe = tl.load(q_pe_ptrs, mask=kpe_mask, other=0.0).to(tl.float32)

#Online softmax state - use large negative value
    NEG_INF = -1e9
    max_val = NEG_INF
    sum_exp = 0.0

#Accumulator for output
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

#Process all TOPK entries
    for k_idx in range(TOPK):
#Load sparse index
        sparse_idx = tl.load(indices_ptr + b_idx * stride_idx_b + k_idx * stride_idx_k)

#Handle padding(-1 means invalid entry)
        valid = sparse_idx >= 0

#Compute page and offset(only used when valid)
        page_idx = tl.where(valid, sparse_idx // PAGE_SIZE, 0)
        page_offset = tl.where(valid, sparse_idx % PAGE_SIZE, 0)

#Load K vectors with MASK to prevent out - of - bounds access
        k_ckv_ptrs = ckv_ptr + page_idx * stride_ckv_p + page_offset * stride_ckv_s + offs_ckv * stride_ckv_d
        k_ckv = tl.load(k_ckv_ptrs, mask=valid & ckv_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        
        k_kpe_ptrs = kpe_ptr + page_idx * stride_kpe_p + page_offset * stride_kpe_s + offs_kpe * stride_kpe_d
        k_kpe = tl.load(k_kpe_ptrs, mask=valid & kpe_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

#Compute logit : (q_nope · k_ckv) + (q_pe · k_kpe)
        dot_ckv = tl.sum(q_nope * k_ckv, axis=0)
        dot_kpe = tl.sum(q_pe * k_kpe, axis=0)

#Compute scaled logit, use NEG_INF for invalid entries
        logit = tl.where(valid, (dot_ckv + dot_kpe) * SM_SCALE, NEG_INF)

#Online softmax update with numerical stability
        new_max = tl.maximum(max_val, logit)

#Compute rescale factor safely
#When max_val is NEG_INF, max_val - new_max could be problematic
#But exp(NEG_INF - anything_finite) = 0, which is correct
        rescale = tl.exp(max_val - new_max)

#Update accumulator with rescaling
        acc = acc * rescale

#Compute weight - 0 for invalid entries
        weight = tl.where(valid, tl.exp(logit - new_max), 0.0)

#Update sum_exp
        sum_exp = sum_exp * rescale + weight

#Accumulate weighted K
        acc = acc + weight * k_ckv

#Update max
        max_val = new_max

#Safe normalization - handle case when all entries are invalid
    out = tl.where(sum_exp > 0, acc / sum_exp, tl.zeros([HEAD_DIM_CKV], dtype=tl.float32))

#Store output as bfloat16 with mask
    out_ptrs = output_ptr + b_idx * stride_out_b + h_idx * stride_out_h + offs_ckv * stride_out_d
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=ckv_mask)

#LSE in log2 base : log2(sum_exp) + max_val / ln(2)
    ln2 = 0.6931471805599453
    lse_val = tl.where(
        sum_exp > 0,
        (max_val + tl.log(sum_exp)) / ln2,
        NEG_INF
    )
    tl.store(lse_ptr + b_idx * stride_lse_b + h_idx * stride_lse_h, lse_val)


def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSA Sparse Attention kernel.
    
    Args:
        q_nope: [num_tokens, num_qo_heads, head_dim_ckv] - Query without positional encoding
        q_pe: [num_tokens, num_qo_heads, head_dim_kpe] - Query positional encoding
        ckv_cache: [num_pages, page_size, head_dim_ckv] - Compressed KV cache
        kpe_cache: [num_pages, page_size, head_dim_kpe] - Key positional encoding cache
        sparse_indices: [num_tokens, topk] - Sparse indices for top-K selection
        sm_scale: Softmax scale factor
    
    Returns:
        output: [num_tokens, num_qo_heads, head_dim_ckv] - Attention output
        lse: [num_tokens, num_qo_heads] - Log-sum-exp in log2 base
    """
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    
    device = q_nope.device

#Allocate outputs
    output = torch.empty((num_tokens, num_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
    
    grid = (num_tokens, num_heads)

#Launch kernel with optimized settings for B200
    _dsa_sparse_attention_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output, lse,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        PAGE_SIZE=page_size,
        SM_SCALE=sm_scale,
        TOPK=topk,
        num_warps=4,
        num_stages=2,
    )
    
    return output, lse
