#!/usr/bin/env python3
"""
快速测试 kernel_1.py
"""
import torch
import sys
from pathlib import Path

# 添加 solution 目录到路径
solution_dir = Path(__file__).parent / "solution" / "triton"
sys.path.insert(0, str(solution_dir))

# 导入 kernel_1
from kernel_1 import kernel as dsa_kernel

def test_kernel_1():
    """测试 kernel_1.py"""

    print("="*80)
    print("🚀 测试 kernel_1.py")
    print("="*80)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: 需要 CUDA 支持")
        return False

    print(f"\n🔧 GPU: {torch.cuda.get_device_name()}")
    print(f"   计算能力: {torch.cuda.get_device_capability()}")

    # 测试参数
    num_tokens = 16
    num_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 256
    page_size = 64
    topk = 2048

    print(f"\n📊 测试参数:")
    print(f"   Tokens: {num_tokens}")
    print(f"   Heads: {num_heads}")
    print(f"   Head Dim CKV: {head_dim_ckv}")
    print(f"   Head Dim KPE: {head_dim_kpe}")
    print(f"   Pages: {num_pages}")
    print(f"   Page Size: {page_size}")
    print(f"   TopK: {topk}")

    device = torch.device('cuda')

    # 创建随机输入数据
    print(f"\n📦 创建测试数据...")
    q_nope = torch.randn(num_tokens, num_heads, head_dim_ckv,
                        dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_heads, head_dim_kpe,
                      dtype=torch.bfloat16, device=device)

    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv,
                           dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe,
                           dtype=torch.bfloat16, device=device)

    # 创建稀疏索引
    max_indices = num_pages * page_size
    sparse_indices = torch.randint(0, max_indices,
                                  (num_tokens, topk),
                                  dtype=torch.int32, device=device)

    # 随机设置一些为 -1 (padding)
    mask = torch.rand(num_tokens, topk, device=device) > 0.95
    sparse_indices[mask] = -1

    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)

    # 准备输出缓冲区
    output = torch.empty_like(q_nope)
    lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    print(f"\n🔥 运行 kernel...")
    try:
        # 运行 kernel
        dsa_kernel(
            q_nope, q_pe,
            ckv_cache, kpe_cache,
            sparse_indices,
            sm_scale,
            output, lse
        )
        torch.cuda.synchronize()

        print(f"✅ Kernel 执行成功！")
        print(f"\n📊 输出统计:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output min: {output.min().item():.4f}")
        print(f"   Output max: {output.max().item():.4f}")
        print(f"   Output mean: {output.mean().item():.4f}")
        print(f"   LSE shape: {lse.shape}")
        print(f"   LSE min: {lse.min().item():.4f}")
        print(f"   LSE max: {lse.max().item():.4f}")
        print(f"   LSE mean: {lse.mean().item():.4f}")

        # 检查是否有 NaN 或 Inf
        if torch.isnan(output).any():
            print(f"\n⚠️  警告: Output 包含 NaN!")
        if torch.isinf(output).any():
            print(f"\n⚠️  警告: Output 包含 Inf!")
        if torch.isnan(lse).any():
            print(f"\n⚠️  警告: LSE 包含 NaN!")
        if torch.isinf(lse).any():
            print(f"   LSE 包含 Inf (这可能是正常的，如果所有索引都无效)")

        # 简单性能测试
        print(f"\n⚡ 性能测试...")
        num_warmup = 3
        num_runs = 10

        for _ in range(num_warmup):
            dsa_kernel(
                q_nope, q_pe,
                ckv_cache, kpe_cache,
                sparse_indices,
                sm_scale,
                output, lse
            )
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(num_runs):
            dsa_kernel(
                q_nope, q_pe,
                ckv_cache, kpe_cache,
                sparse_indices,
                sm_scale,
                output, lse
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / num_runs * 1000
        print(f"   平均时间: {avg_time_ms:.3f} ms")
        print(f"   吞吐量: {num_tokens * num_heads / (avg_time_ms / 1000):.0f} token-heads/sec")

        return True

    except Exception as e:
        print(f"\n❌ Kernel 执行失败！")
        print(f"   错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kernel_1()
    sys.exit(0 if success else 1)
