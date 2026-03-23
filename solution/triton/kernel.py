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
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    offs_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_kpe = tl.arange(0, HEAD_DIM_KPE)

    ckv_mask = offs_ckv < HEAD_DIM_CKV
    kpe_mask = offs_kpe < HEAD_DIM_KPE

    q_nope_ptrs = q_nope_ptr + b_idx * stride_qn_b + h_idx * stride_qn_h + offs_ckv * stride_qn_d
    q_nope = tl.load(q_nope_ptrs, mask=ckv_mask, other=0.0).to(tl.float32)

    q_pe_ptrs = q_pe_ptr + b_idx * stride_qp_b + h_idx * stride_qp_h + offs_kpe * stride_qp_d
    q_pe = tl.load(q_pe_ptrs, mask=kpe_mask, other=0.0).to(tl.float32)

    NEG_INF = -1e9
    max_val = NEG_INF
    sum_exp = 0.0
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    idx_base = indices_ptr + b_idx * stride_idx_b

    for k_start in range(0, TOPK, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        k_idx = k_start + offs_k
        k_mask = k_idx < TOPK

        sparse_idx = tl.load(idx_base + k_idx * stride_idx_k, mask=k_mask, other=-1)
        valid = (sparse_idx >= 0) & k_mask

        safe_idx = tl.where(valid, sparse_idx, 0)
        page_idx = safe_idx // PAGE_SIZE
        page_offset = safe_idx % PAGE_SIZE

        load_mask_ckv = valid[:, None] & ckv_mask[None, :]
        load_mask_kpe = valid[:, None] & kpe_mask[None, :]

        k_ckv_ptrs = ckv_ptr + page_idx[:, None] * stride_ckv_p + page_offset[:, None] * stride_ckv_s + offs_ckv[None, :] * stride_ckv_d
        k_ckv = tl.load(k_ckv_ptrs, mask=load_mask_ckv, other=0.0, eviction_policy="evict_last").to(tl.float32)

        k_kpe_ptrs = kpe_ptr + page_idx[:, None] * stride_kpe_p + page_offset[:, None] * stride_kpe_s + offs_kpe[None, :] * stride_kpe_d
        k_kpe = tl.load(k_kpe_ptrs, mask=load_mask_kpe, other=0.0, eviction_policy="evict_last").to(tl.float32)

        dot_ckv = tl.sum(q_nope[None, :] * k_ckv, axis=1)
        dot_kpe = tl.sum(q_pe[None, :] * k_kpe, axis=1)

        logit = (dot_ckv + dot_kpe) * SM_SCALE
        logit = tl.where(valid, logit, NEG_INF)

        block_max = tl.max(logit, axis=0)
        new_max = tl.maximum(max_val, block_max)

        rescale = tl.exp(max_val - new_max)
        acc = acc * rescale

        weight = tl.where(valid, tl.exp(logit - new_max), 0.0)

        sum_exp = sum_exp * rescale + tl.sum(weight, axis=0)

        acc = acc + tl.sum(weight[:, None] * k_ckv, axis=0)

        max_val = new_max

    out = tl.where(sum_exp > 0, acc / sum_exp, 0.0)

    out_ptrs = output_ptr + b_idx * stride_out_b + h_idx * stride_out_h + offs_ckv * stride_out_d
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=ckv_mask)

    ln2 = 0.6931471805599453
    lse_val = tl.where(sum_exp > 0, (max_val + tl.log(sum_exp)) / ln2, NEG_INF)
    tl.store(lse_ptr + b_idx * stride_lse_b + h_idx * stride_lse_h, lse_val)


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
    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert output.shape == (num_tokens, num_heads, head_dim_ckv), "Output shape mismatch"
    assert lse.shape == (num_tokens, num_heads), "LSE shape mismatch"

    grid = (num_tokens, num_heads)

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
        BLOCK_K=8,
        num_warps=4,
        num_stages=4,
    )
