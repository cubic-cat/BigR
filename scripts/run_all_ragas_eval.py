#!/usr/bin/env python3
"""自动化执行完整的语义QA RAGAS评估流程。

用途
-----
依次执行：
  1. 评估 Dense 检索
  2. 评估 BM25 + CrossEncoder 重排
  3. 生成评估报告

使用方法
--------
python scripts/run_all_ragas_eval.py

注意：预计运行时间约4-6小时，请确保：
  - Qdrant服务正在运行
  - .env文件已正确配置LLM API密钥
  - 已安装所有依赖（包括ragas）
"""

import subprocess
import sys
import time
from pathlib import Path
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用官方推荐的镜像源

_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: str, desc: str) -> bool:
    """执行命令并显示实时输出。"""
    print(f"\n{'='*60}")
    print(f"[开始] {desc}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    print("-" * 60)

    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=_ROOT,
            capture_output=False,
            text=True,
            encoding="utf-8",
        )
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"完成时间: {elapsed:.1f}秒")
        
        if result.returncode != 0:
            print(f"[错误] 命令执行失败，返回码: {result.returncode}")
            return False
        print(f"[成功] {desc}")
        return True
        
    except Exception as e:
        print(f"[错误] 执行命令时发生异常: {e}")
        return False


def main():
    print("="*60)
    print("语义QA RAGAS评估 - 自动化执行脚本")
    print("="*60)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n警告：此脚本需要较长时间运行（约4-6小时）")
    print("建议在后台或夜间运行\n")

    # 步骤1: 评估Dense检索
    success = run_command(
        "python scripts/run_ragas_evaluation.py --retrieval_method dense",
        "评估 Dense 检索"
    )
    if not success:
        print("\n[终止] Dense检索评估失败，中止执行")
        sys.exit(1)

    # 步骤2: 评估BM25 + CrossEncoder重排
    success = run_command(
        "python scripts/run_ragas_evaluation.py --retrieval_method bm25+cross_encoder",
        "评估 BM25 + CrossEncoder 重排"
    )
    if not success:
        print("\n[终止] BM25+CrossEncoder评估失败，中止执行")
        sys.exit(1)

    # 步骤3: 生成评估报告
    success = run_command(
        "python scripts/generate_eval_report.py",
        "生成评估报告"
    )
    if not success:
        print("\n[终止] 报告生成失败，中止执行")
        sys.exit(1)

    # 完成
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n生成的文件:")
    print("  - outputs/ragas_eval/results_dense.json")
    print("  - outputs/ragas_eval/summary_dense.json")
    print("  - outputs/ragas_eval/results_bm25_cross_encoder.json")
    print("  - outputs/ragas_eval/summary_bm25_cross_encoder.json")
    print("  - outputs/reports/ragas_eval_report.md")
    print("\n请查看报告了解评估结果分析。")


if __name__ == "__main__":
    main()
