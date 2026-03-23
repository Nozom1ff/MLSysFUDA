#!/usr/bin/env python3
"""
轻量级本地 Kernel 测试工具

无需依赖完整的 flashinfer-bench 框架，直接从 workload 文件加载测试数据。
适用于快速开发和验证 kernel 实现。

使用方法:
    python test_kernel_local.py                          # 使用默认配置
    python test_kernel_local.py --num-tokens 64          # 自定义参数
    python test_kernel_local.py --workload-id xxx        # 使用特定 workload
    python test_kernel_local.py --quick                  # 快速测试（减少迭代）
"""

import torch
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# 添加 solution 目录到路径
solution_dir = Path(__file__).parent / "solution" / "triton"
sys.path.insert(0, str(solution_dir))

from kernel import kernel as dsa_kernel


class WorkloadLoader:
    """从 FlashInfer-Bench workload 文件加载测试数据"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.workload_path = self.dataset_path / "workloads" / "dsa_paged"

    def load_workload(self, workload_id: str) -> Dict[str, Any]:
        """加载指定的 workload"""
        workload_file = self.workload_path / f"{workload_id}.jsonl"

        if not workload_file.exists():
            print(f"❌ Workload 文件不存在: {workload_file}")
            print(f"可用的 workloads:")
            self.list_workloads()
            return None

        print(f"📂 加载 workload: {workload_id}")
        print(f"   文件: {workload_file}")

        with open(workload_file, 'r') as f:
            workload = json.loads(f.readline())

        return workload

    def list_workloads(self):
        """列出所有可用的 workloads"""
        if not self.workload_path.exists():
            print(f"❌ Workload 路径不存在: {self.workload_path}")
            return

        workload_files = list(self.workload_path.glob("dsa_sparse_attention_*.jsonl"))
        for wf in workload_files[:10]:  # 只显示前 10 个
            print(f"   - {wf.stem}")

        if len(workload_files) > 10:
            print(f"   ... 还有 {len(workload_files) - 10} 个")


class LocalKernelTester:
    """本地 kernel 测试器"""

    def __init__(self, workload: Dict[str, Any]):
        self.workload = workload
        self.inputs = workload['inputs']
        self.expected_output = workload['output']

        # 从 inputs 提取数据
        self.num_tokens = self._get_shape(self.inputs['q_nope'])[0]
        self.num_heads = self._get_shape(self.inputs['q_nope'])[1]
        self.head_dim_ckv = self._get_shape(self.inputs['q_nope'])[2]
        self.head_dim_kpe = self._get_shape(self.inputs['q_pe'])[2]
        self.num_pages = self._get_shape(self.inputs['ckv_cache'])[0]
        self.page_size = self._get_shape(self.inputs['ckv_cache'])[1]
        self.topk = self._get_shape(self.inputs['sparse_indices'])[1]
        self.sm_scale = self.inputs['sm_scale']

        print(f"\n📊 Workload 参数:")
        print(f"   - Tokens: {self.num_tokens}")
        print(f"   - Heads: {self.num_heads}")
        print(f"   - Head Dim CKV: {self.head_dim_ckv}")
        print(f"   - Head Dim KPE: {self.head_dim_kpe}")
        print(f"   - Pages: {self.num_pages}")
        print(f"   - Page Size: {self.page_size}")
        print(f"   - TopK: {self.topk}")

    def _get_shape(self, tensor_data) -> list:
        """从 tensor 数据获取 shape"""
        # 如果直接是 tensor，返回 shape
        if isinstance(tensor_data, torch.Tensor):
            return list(tensor_data.shape)
        # 如果是字典格式（来自真实 workload）
        if isinstance(tensor_data, dict):
            if 'shape' in tensor_data:
                return tensor_data['shape']
            if 'tensor' in tensor_data:
                return list(tensor_data['tensor'].shape)
        # 默认尝试作为 tensor 处理
        return list(tensor_data.shape)

    def load_tensors_from_workload(self) -> Tuple[Dict, Dict]:
        """从 workload 加载实际的 tensor 数据"""

        def parse_tensor(data):
            """从字典数据创建 torch tensor"""
            if isinstance(data, torch.Tensor):
                return data

            # 如果是字典格式，包含 values 和 shape
            if isinstance(data, dict) and 'tensor' in data:
                tensor_data = data['tensor']
                return tensor_data

            # 如果是简单字典，包含 shape 和 values
            if isinstance(data, dict) and 'shape' in data and 'values' in data:
                values = data['values']
                shape = data['shape']
                dtype = data.get('dtype', 'float32')

                # 转换 dtype
                if dtype == 'bfloat16':
                    torch_dtype = torch.bfloat16
                elif dtype == 'float32':
                    torch_dtype = torch.float32
                elif dtype == 'int32':
                    torch_dtype = torch.int32
                else:
                    torch_dtype = torch.float32

                # 从列表创建 tensor
                if isinstance(values, list):
                    # 扁平化然后 reshape
                    flat_values = []
                    for item in values:
                        if isinstance(item, list):
                            flat_values.extend(item)
                        else:
                            flat_values.append(item)
                    tensor = torch.tensor(flat_values, dtype=torch_dtype).view(shape)
                else:
                    tensor = torch.tensor(values, dtype=torch_dtype)

                return tensor

            return data

        # 加载所有输入
        inputs = {}
        for key, value in self.inputs.items():
            parsed = parse_tensor(value)
            # 如果是 tensor，移到 GPU；如果是标量，保持不变
            if isinstance(parsed, torch.Tensor):
                inputs[key] = parsed.cuda()
            else:
                inputs[key] = parsed

        # 加载期望输出
        expected = {}
        for key, value in self.expected_output.items():
            parsed = parse_tensor(value)
            if isinstance(parsed, torch.Tensor):
                expected[key] = parsed.cuda()
            else:
                expected[key] = parsed

        return inputs, expected

    def test_kernel(self, inputs: Dict, expected: Dict,
                   num_warmup: int = 5, num_runs: int = 50) -> Dict[str, Any]:
        """测试 kernel 性能和正确性"""

        # 准备输出缓冲区
        output = torch.empty_like(inputs['q_nope'])
        lse = torch.empty(
            inputs['q_nope'].shape[0],
            inputs['q_nope'].shape[1],
            dtype=torch.float32,
            device='cuda'
        )

        print(f"\n🔥 测试配置:")
        print(f"   - Warmup: {num_warmup} 次")
        print(f"   - Runs: {num_runs} 次")

        # Warmup
        print(f"\n⏳ 预热中...")
        for _ in range(num_warmup):
            dsa_kernel(
                inputs['q_nope'], inputs['q_pe'],
                inputs['ckv_cache'], inputs['kpe_cache'],
                inputs['sparse_indices'],
                self.sm_scale,
                output, lse
            )
        torch.cuda.synchronize()

        # Benchmark
        print(f"🚀 性能测试中...")
        start_time = time.perf_counter()
        for _ in range(num_runs):
            dsa_kernel(
                inputs['q_nope'], inputs['q_pe'],
                inputs['ckv_cache'], inputs['kpe_cache'],
                inputs['sparse_indices'],
                self.sm_scale,
                output, lse
            )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) / num_runs * 1000

        # 验证正确性
        print(f"\n✅ 验证正确性...")
        output_diff = torch.abs(output - expected['output']).max().item()
        lse_diff = torch.abs(lse - expected['lse']).max().item()

        output_match = torch.allclose(output, expected['output'], rtol=1e-3, atol=1e-3)
        lse_match = torch.allclose(lse, expected['lse'], rtol=1e-3, atol=1e-3)

        print(f"   Output 最大差异: {output_diff:.6f}")
        print(f"   LSE 最大差异: {lse_diff:.6f}")
        print(f"   Output 匹配: {'✅' if output_match else '❌'}")
        print(f"   LSE 匹配: {'✅' if lse_match else '❌'}")

        # 返回结果
        results = {
            'avg_time_ms': avg_time_ms,
            'throughput': (self.num_tokens * self.num_heads) / (avg_time_ms / 1000),
            'output_match': output_match,
            'lse_match': lse_match,
            'output_diff': output_diff,
            'lse_diff': lse_diff,
            'all_match': output_match and lse_match
        }

        return results


def create_synthetic_workload(
    num_tokens: int = 32,
    num_heads: int = 16,
    head_dim_ckv: int = 512,
    head_dim_kpe: int = 64,
    num_pages: int = 1024,
    page_size: int = 64,
    topk: int = 2048
) -> Dict[str, Any]:
    """创建合成的测试 workload（无需外部数据）"""

    print(f"\n🔨 创建合成测试数据")
    print(f"   Tokens: {num_tokens}, Heads: {num_heads}")
    print(f"   Head Dim CKV: {head_dim_ckv}, KPE: {head_dim_kpe}")
    print(f"   Pages: {num_pages}, Page Size: {page_size}")
    print(f"   TopK: {topk}")

    device = torch.device('cuda')

    # 创建随机输入数据
    q_nope = torch.randn(num_tokens, num_heads, head_dim_ckv,
                        dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_heads, head_dim_kpe,
                      dtype=torch.bfloat16, device=device)

    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv,
                           dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe,
                           dtype=torch.bfloat16, device=device)

    # 创建稀疏索引（随机选择，有些用 -1 padding）
    max_indices = num_pages * page_size
    sparse_indices = torch.randint(0, max_indices,
                                  (num_tokens, topk),
                                  dtype=torch.int32, device=device)

    # 随机设置一些为 -1 (padding)
    mask = torch.rand(num_tokens, topk, device=device) > 0.95
    sparse_indices[mask] = -1

    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)

    # 创建参考输出（使用简单的 PyTorch 实现）
    print(f"   计算参考输出...")
    output_ref, lse_ref = compute_reference_output(
        q_nope, q_pe, ckv_cache, kpe_cache,
        sparse_indices, sm_scale, page_size
    )

    # 构造 workload 格式
    workload = {
        'inputs': {
            'q_nope': q_nope,
            'q_pe': q_pe,
            'ckv_cache': ckv_cache,
            'kpe_cache': kpe_cache,
            'sparse_indices': sparse_indices,
            'sm_scale': sm_scale
        },
        'output': {
            'output': output_ref,
            'lse': lse_ref
        }
    }

    return workload


@torch.no_grad()
def compute_reference_output(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
    page_size: int
):
    """计算参考输出（使用 PyTorch）"""

    import math

    num_tokens, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    device = q_nope.device

    # Flatten paged KV cache
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).float()
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).float()

    output = torch.zeros(
        (num_tokens, num_heads, head_dim_ckv),
        dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (num_tokens, num_heads),
        -float('inf'),
        dtype=torch.float32, device=device
    )

    for t in range(num_tokens):
        indices = sparse_indices[t]

        # 处理 padding
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.long()
        Kc = Kc_all[tok_idx]
        Kp = Kp_all[tok_idx]
        qn = q_nope[t].float()
        qp = q_pe[t].float()

        # 计算注意力
        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        # LSE (log2 base)
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Attention output
        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[t] = out.to(torch.bfloat16)

    return output, lse


def print_results(results: Dict[str, Any]):
    """打印测试结果"""
    print(f"\n{'='*80}")
    print(f"📊 测试结果")
    print(f"{'='*80}")

    print(f"\n⚡ 性能:")
    print(f"   平均时间: {results['avg_time_ms']:.3f} ms")
    print(f"   吞吐量: {results['throughput']:.0f} token-heads/sec")
    print(f"   每次 token: {results['avg_time_ms'] / results.get('num_tokens', 1) * 1000:.3f} μs")

    print(f"\n✅ 正确性:")
    status = '✅ PASS' if results['all_match'] else '❌ FAIL'
    print(f"   状态: {status}")
    print(f"   Output 最大误差: {results['output_diff']:.6e}")
    print(f"   LSE 最大误差: {results['lse_diff']:.6e}")

    if results['all_match']:
        print(f"\n🎉 恭喜！Kernel 通过测试！")
    else:
        print(f"\n⚠️  Kernel 输出与参考不匹配，请检查实现")


def main():
    parser = argparse.ArgumentParser(description="DSA Kernel 本地测试工具")
    parser.add_argument("--dataset", type=str,
                       default="../mlsys26-contest",
                       help="FlashInfer 数据集路径")
    parser.add_argument("--workload-id", type=str, default=None,
                       help="指定 workload ID（如: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64）")
    parser.add_argument("--num-tokens", type=int, default=32,
                       help="合成数据的 token 数量")
    parser.add_argument("--num-heads", type=int, default=16,
                       help="合成数据的 head 数量")
    parser.add_argument("--head-dim-ckv", type=int, default=512,
                       help="合成数据的 CKV head 维度")
    parser.add_argument("--head-dim-kpe", type=int, default=64,
                       help="合成数据的 KPE head 维度")
    parser.add_argument("--topk", type=int, default=2048,
                       help="合成数据的 TopK 值")
    parser.add_argument("--num-warmup", type=int, default=5,
                       help="预热次数")
    parser.add_argument("--num-runs", type=int, default=50,
                       help="性能测试运行次数")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式（减少迭代）")

    args = parser.parse_args()

    if args.quick:
        args.num_warmup = 2
        args.num_runs = 10

    print("="*80)
    print("🚀 DSA Kernel 本地测试工具")
    print("="*80)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: 需要 CUDA 支持")
        return

    print(f"\n🔧 GPU: {torch.cuda.get_device_name()}")
    print(f"   计算能力: {torch.cuda.get_device_capability()}")

    # 加载或创建 workload
    workload = None
    tester = None

    if args.workload_id:
        # 从数据集加载
        dataset_path = args.dataset
        if not Path(dataset_path).exists():
            print(f"❌ 数据集路径不存在: {dataset_path}")
            print(f"   使用 --synthetic 模式创建合成测试数据")
            args.workload_id = None
        else:
            loader = WorkloadLoader(dataset_path)
            workload = loader.load_workload(args.workload_id)

            if workload:
                tester = LocalKernelTester(workload)

    if not args.workload_id or workload is None:
        # 创建合成数据
        workload = create_synthetic_workload(
            num_tokens=args.num_tokens,
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            topk=args.topk
        )
        tester = LocalKernelTester(workload)

    # 加载 tensor 数据
    print(f"\n📦 准备测试数据...")
    inputs, expected = tester.load_tensors_from_workload()

    # 测试 kernel
    results = tester.test_kernel(inputs, expected, args.num_warmup, args.num_runs)
    results['num_tokens'] = inputs['q_nope'].shape[0]

    # 打印结果
    print_results(results)

    # 返回状态码
    return 0 if results['all_match'] else 1


if __name__ == "__main__":
    exit(main())
