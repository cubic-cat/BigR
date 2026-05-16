#!/usr/bin/env python3
"""生成RAGAS评估结果的可视化图表。

用途
-----
根据outputs/ragas_eval/目录下的summary_*.json文件生成对比图表，
用于PPT展示。

使用方法
--------
python scripts/generate_ragas_charts.py

输出
-----
output/charts/ragas_comparison.png - RAGAS指标对比柱状图
output/charts/ragas_radar.png     - 雷达图对比
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = _ROOT / "outputs" / "ragas_eval"
CHARTS_DIR = _ROOT / "output" / "charts"

METRIC_LABELS = {
    "context_precision": "Context\nPrecision",
    "context_recall": "Context\nRecall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer\nRelevancy",
}

METHOD_COLORS = {
    "bm25": "#4C78A8",
    "dense": "#F58518",
    "bm25+cross_encoder": "#72B7BC",
}


def load_results():
    """加载所有摘要文件。"""
    results = {}
    for f in RESULTS_DIR.glob("summary_*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            method = data["method"]
            results[method] = data
    return results


def generate_bar_chart(results: dict, output_path: Path):
    """生成RAGAS指标对比柱状图。"""
    methods = list(results.keys())
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    for i, method in enumerate(methods):
        values = [results[method]["metrics"].get(m, {}).get("mean", 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=METHOD_COLORS.get(method, f"C{i}"))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("RAGAS Metrics Comparison on Semantic QA Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"已保存柱状图: {output_path}")


def generate_radar_chart(results: dict, output_path: Path):
    """生成雷达图对比。"""
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    n_metrics = len(metrics)
    
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"), dpi=150)
    
    for method, data in results.items():
        values = [data["metrics"].get(m, {}).get("mean", 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=METHOD_COLORS.get(method, "blue"))
        ax.fill(angles, values, alpha=0.15, color=METHOD_COLORS.get(method, "blue"))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS Metrics Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"已保存雷达图: {output_path}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    if not results:
        print(f"[错误] 未找到结果文件: {RESULTS_DIR}/summary_*.json")
        sys.exit(1)
    
    print(f"已加载 {len(results)} 个方法的结果")
    for method in results:
        print(f"  - {method}")
    
    generate_bar_chart(results, CHARTS_DIR / "ragas_comparison.png")
    generate_radar_chart(results, CHARTS_DIR / "ragas_radar.png")
    
    print("\n图表生成完成！")
    print(f"图表保存位置: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
