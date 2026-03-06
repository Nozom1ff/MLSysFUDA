import torch
import triton
import triton.language as tl
import math

# 定义常量以提高可读性
BLOCK_N = 64  # 每个循环块处理的KV token数量
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOPK = 2048
NUM_HEADS = 16


import math
import torch


@torch.no_grad()
def torch_ref(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Squeeze page dimension (page_size=1)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse


@triton.jit
def _kernel_dsa_sparse(
    # Pointers
    Q_Nope_ptr, Q_Pe_ptr,
    CKV_ptr, KPE_ptr,
    Indices_ptr,
    Out_ptr, Lse_ptr,
    # Strides
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_ps, stride_ckv_d,
    stride_kpe_p, stride_kpe_ps, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_h, stride_out_d,
    stride_lse_t, stride_lse_h,
    # Scalars
    sm_scale,
    # Consts
    BLOCK_N: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr
):
    # Program ID setup
    pid = tl.program_id(0)
    cur_head_idx = pid % 16
    cur_token_idx = pid // 16

    # Dimension offsets
    offs_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_kpe = tl.arange(0, HEAD_DIM_KPE)

    # 1. Load Queries
    # Q_nope: [num_tokens, 16, 512]
    qn_ptr = Q_Nope_ptr + (cur_token_idx * stride_qn_t) + (cur_head_idx * stride_qn_h) + (offs_ckv * stride_qn_d)
    q_nope = tl.load(qn_ptr).to(tl.float32)

    # Q_pe: [num_tokens, 16, 64]
    qp_ptr = Q_Pe_ptr + (cur_token_idx * stride_qp_t) + (cur_head_idx * stride_qp_h) + (offs_kpe * stride_qp_d)
    q_pe = tl.load(qp_ptr).to(tl.float32)

    # 2. Online Softmax Accumulators
    # m_i: max logit (initialized to -inf)
    # l_i: sum exp
    # acc: weighted sum
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    # 3. Loop over TopK indices
    for start_n in range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load Sparse Indices: [num_tokens, 2048]
        idx_ptr = Indices_ptr + (cur_token_idx * stride_idx_t) + (offs_n * stride_idx_k)
        indices = tl.load(idx_ptr)

        # Handle padding (-1)
        mask_valid = indices != -1

        # Indirect Memory Access (Gather from Page Cache)
        # CKV: [num_pages, 1, 512]
        # Ptr = base + page_idx * stride_p + offset * stride_d
        ckv_loc = (indices[:, None] * stride_ckv_p) + (offs_ckv[None, :] * stride_ckv_d)
        kpe_loc = (indices[:, None] * stride_kpe_p) + (offs_kpe[None, :] * stride_kpe_d)
        
        ckv_ptrs = CKV_ptr + ckv_loc
        kpe_ptrs = KPE_ptr + kpe_loc

        # Load KVs with masking for invalid pages
        # mask needs to be broadcasted to [BLOCK_N, HEAD_DIM]
        kc = tl.load(ckv_ptrs, mask=mask_valid[:, None], other=0.0).to(tl.float32)
        kp = tl.load(kpe_ptrs, mask=mask_valid[:, None], other=0.0).to(tl.float32)

        # Compute Attention Scores: (Qn * Kc) + (Qp * Kp)
        # q_nope[None, :] broadcasts [512] -> [1, 512]
        # kc is [BLOCK_N, 512]
        # result of multiplication is [BLOCK_N, 512]
        # sum(axis=1) -> [BLOCK_N]
        score_n = tl.sum(q_nope[None, :] * kc, axis=1)
        score_p = tl.sum(q_pe[None, :] * kp, axis=1)
        
        scores = (score_n + score_p) * sm_scale
        
        # Mask padded scores to -inf
        scores = tl.where(mask_valid, scores, -float("inf"))

        # Online Softmax Update
        # 1. Compute max of current block
        m_curr = tl.max(scores, 0)
        
        # 2. Update global max
        # FIX: Use tl.maximum for element-wise max between scalar accumulator and new scalar
        m_new = tl.maximum(m_i, m_curr)
        
        # 3. Compute Exponentials
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        
        # 4. Update Denominator
        l_i = l_i * alpha + tl.sum(p, 0)
        
        # 5. Update Numerator (Accumulator)
        # acc = acc * alpha + p @ kc
        # p[:, None] -> [BLOCK_N, 1], kc -> [BLOCK_N, 512]
        weighted_v = tl.sum(p[:, None] * kc, axis=0)
        acc = acc * alpha + weighted_v
        
        m_i = m_new

    # 4. Finalize Output
    out = acc / l_i
    
    # Store Output
    out_off = (cur_token_idx * stride_out_t) + (cur_head_idx * stride_out_h) + (offs_ckv * stride_out_d)
    tl.store(Out_ptr + out_off, out.to(tl.bfloat16))

    # Compute and Store LSE (Base 2)
    # Result = log2(sum(e^x)) = (log(l_i) + m_i) * log2(e)
    LOG2_E = 1.44269504
    lse_val = (tl.log(l_i) + m_i) * LOG2_E
    
    lse_off = (cur_token_idx * stride_lse_t) + (cur_head_idx * stride_lse_h)
    tl.store(Lse_ptr + lse_off, lse_val.to(tl.float32))


def dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
):
    """
    Batched Native Sparse Attention (DSA) with sparse TopK KV cache selection.
    Arguments match the definition inputs exactly.
    """
    # 1. Check Dimensions & Constraints
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16, f"num_qo_heads must be 16, got {num_qo_heads}"
    assert head_dim_ckv == 512, f"head_dim_ckv must be 512, got {head_dim_ckv}"
    assert head_dim_kpe == 64, f"head_dim_kpe must be 64, got {head_dim_kpe}"
    assert page_size == 1, f"page_size must be 1, got {page_size}"
    assert topk == 2048, f"topk must be 2048, got {topk}"

    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # 2. Allocate Outputs
    output = torch.empty(
        (num_tokens, num_qo_heads, head_dim_ckv), 
        dtype=torch.bfloat16, 
        device=device
    )
    lse = torch.empty(
        (num_tokens, num_qo_heads), 
        dtype=torch.float32, 
        device=device
    )

    # 3. Kernel Launch
    grid = (num_tokens * num_qo_heads, 1, 1)
    
    _kernel_dsa_sparse[grid](
        q_nope, q_pe,
        ckv_cache, kpe_cache,
        sparse_indices,
        output, lse,
        # Strides
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        # Scalar
        sm_scale,
        # Constants
        BLOCK_N=BLOCK_N,
        HEAD_DIM_CKV=HEAD_DIM_CKV,
        HEAD_DIM_KPE=HEAD_DIM_KPE,
        TOPK=TOPK
    )

    return output, lse
# --- 验证代码 (用于测试正确性) ---
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Test Config
    num_tokens = 4
    num_heads = 16
    d_ckv = 512
    d_kpe = 64
    topk = 2048
    num_pages = 5000
    
    # Inputs
    q_nope = torch.randn(num_tokens, num_heads, d_ckv, device='cuda', dtype=torch.bfloat16)
    q_pe = torch.randn(num_tokens, num_heads, d_kpe, device='cuda', dtype=torch.bfloat16)
    
    # Cache (Page Size = 1)
    ckv_cache = torch.randn(num_pages, 1, d_ckv, device='cuda', dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, 1, d_kpe, device='cuda', dtype=torch.bfloat16)
    
    # Sparse Indices (Randomly select pages, simulate some -1 padding)
    indices = torch.randint(0, num_pages, (num_tokens, topk), device='cuda', dtype=torch.int32)
    # Mask last few to -1 to test padding logic
    indices[:, -10:] = -1
    
    sm_scale = 1.0 / math.sqrt(d_ckv)

    # Reference Implementation
    from torch.nn.functional import softmax
    
    # Import or define the torch run function provided in the prompt
    # (Here we just assume the 'run' function from the prompt is available or reimplement logic for check)
    def torch_ref(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # ... Copy of the provided torch code ...
        Kc_all = ckv_cache.squeeze(1).float()
        Kp_all = kpe_cache.squeeze(1).float()
        out = torch.zeros_like(q_nope)
        lse_ref = torch.full((num_tokens, num_heads), -float("inf"), device=q_nope.device)
        
        for t in range(num_tokens):
            idx = sparse_indices[t]
            valid = idx != -1
            idx_v = idx[valid].long()
            if len(idx_v) == 0: continue
            
            kc = Kc_all[idx_v]
            kp = Kp_all[idx_v]
            qn = q_nope[t].float()
            qp = q_pe[t].float()
            
            logits = (qn @ kc.T) + (qp @ kp.T)
            logits_scaled = logits * sm_scale
            
            lse_ref[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=-1)
            out[t] = (attn @ kc).to(torch.bfloat16)
        return out, lse_ref

    # Run Reference
    ref_out, ref_lse = torch_ref(q_nope, q_pe, ckv_cache, kpe_cache, indices, sm_scale)
    
    # Run Triton
    tri_out, tri_lse = dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1(
        q_nope, q_pe, ckv_cache, kpe_cache, indices, sm_scale
    )
    
    # Verification
    print(f"Max Diff Output: {(ref_out - tri_out).abs().max().item()}")
    print(f"Max Diff LSE: {(ref_lse - tri_lse).abs().max().item()}")
    
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=1e-2)
    print("Verification Passed!")