#!/usr/bin/env python3
"""从RAGAS评估结果自动生成Markdown评估报告。

用途
-----
读取 run_ragas_evaluation.py 的评估结果并生成简洁的Markdown报告，包括：
  - 总结实验配置
  - 对比不同检索方法的RAGAS指标
  - 分析启发式基准测试是否存在词法偏差
  - 得出关于语义QA下检索方法表现的结论

使用方法
--------
# 默认：从 outputs/ragas_eval/ 读取所有可用结果
    python scripts/generate_eval_report.py

# 显式指定结果文件
    python scripts/generate_eval_report.py --results outputs/ragas_eval/summary_bm25.json outputs/ragas_eval/summary_dense.json

# 自定义输出路径
    python scripts/generate_eval_report.py --output outputs/reports/my_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = "outputs/ragas_eval"
DEFAULT_OUTPUT = "outputs/reports/ragas_eval_report.md"
METRIC_NAMES = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

# 指标中文名称映射
METRIC_CN_NAMES = {
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
}


# ---------------------------------------------------------------------------
# 报告生成函数
# ---------------------------------------------------------------------------

def load_summaries(results_dir: str) -> list[dict]:
    """从结果目录加载所有 summary_*.json 文件。"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[错误] 结果目录不存在: {results_path}")
        sys.exit(1)

    summaries = []
    for f in sorted(results_path.glob("summary_*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            data["_source_file"] = f.name
            summaries.append(data)

    return summaries


def load_summaries_from_files(file_paths: list[str]) -> list[dict]:
    """从指定路径列表加载摘要JSON文件。"""
    summaries = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[警告] 文件不存在: {path}")
            continue
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            data["_source_file"] = path.name
            summaries.append(data)
    return summaries


def generate_method_comparison_table(summaries: list[dict]) -> str:
    """生成RAGAS指标对比的Markdown表格。"""
    if not summaries:
        return "*暂无评估结果可用。*"

    # 构建表格表头
    methods = [s["method"] for s in summaries]
    header = "| 指标 |" + "|".join(f" {m} |" for m in methods)
    separator = "|--------|" + "|".join(["------|" for _ in methods])

    rows = []
    for metric in METRIC_NAMES:
        row = f"| {METRIC_CN_NAMES.get(metric, metric)} |"
        values = []
        for s in summaries:
            val = s.get("metrics", {}).get(metric, {}).get("mean", None)
            if val is not None:
                values.append((val, s["method"]))
                row += f" {val:.4f} |"
            else:
                row += " — |"
        rows.append(row)

    # 添加延迟行
    latency_row = "| 平均检索延迟(ms) |"
    for s in summaries:
        lat = s.get("avg_retrieval_latency_ms", None)
        if lat is not None:
            latency_row += f" {lat:.1f} |"
        else:
            latency_row += " — |"
    rows.append(latency_row)

    return "\n".join([header, separator] + rows)


def generate_best_method_analysis(summaries: list[dict]) -> str:
    """自动分析每个指标的最佳方法。"""
    if not summaries:
        return ""

    lines = []
    for metric in METRIC_NAMES:
        best_method = None
        best_val = -1.0
        for s in summaries:
            val = s.get("metrics", {}).get(metric, {}).get("mean", None)
            if val is not None and val > best_val:
                best_val = val
                best_method = s["method"]
        if best_method:
            metric_cn = METRIC_CN_NAMES.get(metric, metric)
            lines.append(f"- **{metric_cn}**: {best_method} ({best_val:.4f})")

    return "\n".join(lines)


def generate_lexical_bias_analysis(summaries: list[dict], qa_path: str | None) -> str:
    """基于语义QA结果分析词法偏差是否存在。"""
    lines = []

    # 检查是否有BM25和Dense的结果
    methods = {s["method"] for s in summaries}
    has_bm25 = "bm25" in methods
    has_dense = "dense" in methods
    has_rerank = "bm25+cross_encoder" in methods

    if has_bm25 and has_dense:
        bm25_summary = next(s for s in summaries if s["method"] == "bm25")
        dense_summary = next(s for s in summaries if s["method"] == "dense")

        # 比较context_recall（最能反映检索质量）
        bm25_cr = bm25_summary.get("metrics", {}).get("context_recall", {}).get("mean", 0)
        dense_cr = dense_summary.get("metrics", {}).get("context_recall", {}).get("mean", 0)

        lines.append("### 词法偏差分析")
        lines.append("")
        lines.append(f"- BM25 上下文召回率: {bm25_cr:.4f}")
        lines.append(f"- Dense 上下文召回率: {dense_cr:.4f}")

        if bm25_cr > dense_cr * 1.1:
            lines.append("")
            lines.append("**发现**: BM25在语义QA条件下的context_recall仍优于Dense。")
            lines.append("这表明BM25的优势并非完全源于词法偏差——关键词匹配")
            lines.append("即使在查询被转述后仍然有效。")
        elif dense_cr > bm25_cr * 1.1:
            lines.append("")
            lines.append("**发现**: Dense检索在语义QA条件下的context_recall优于BM25！")
            lines.append("这表明启发式基准测试确实引入了有利于BM25的词法偏差。")
            lines.append("在真实语义查询下，Dense检索表现出更强的泛化能力。")
        else:
            lines.append("")
            lines.append("**发现**: BM25和Dense在语义QA条件下的context_recall表现相当。")
            lines.append("启发式基准测试中的词法偏差可能略微夸大了BM25的优势，")
            lines.append("但差异并不显著。")

    if has_bm25 and has_rerank:
        bm25_summary = next(s for s in summaries if s["method"] == "bm25")
        rerank_summary = next(s for s in summaries if s["method"] == "bm25+cross_encoder")

        bm25_cp = bm25_summary.get("metrics", {}).get("context_precision", {}).get("mean", 0)
        rerank_cp = rerank_summary.get("metrics", {}).get("context_precision", {}).get("mean", 0)

        lines.append("")
        lines.append("### 重排效果分析")
        lines.append("")
        lines.append(f"- BM25 上下文精确率: {bm25_cp:.4f}")
        lines.append(f"- BM25+CrossEncoder 上下文精确率: {rerank_cp:.4f}")

        if rerank_cp > bm25_cp * 1.05:
            lines.append("")
            lines.append("**发现**: CrossEncoder重排在上下文精确率方面提供了明显改善。")
            lines.append("在语义QA条件下，重排有助于过滤不相关的上下文。")
        elif rerank_cp > bm25_cp:
            lines.append("")
            lines.append("**发现**: CrossEncoder重排在上下文精确率方面提供了边际改善。")
            lines.append("收益较小，表明BM25的top结果已经相当相关。")
        else:
            lines.append("")
            lines.append("**发现**: CrossEncoder重排未能改善上下文精确率。")
            lines.append("仅使用BM25检索可能已足以满足此基准测试。")

    # 加载QA数据集进行词法重叠分析
    if qa_path:
        qa_file = Path(qa_path)
        if qa_file.exists():
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
            if qa_pairs:
                overlaps = [q.get("lexical_overlap", 0) for q in qa_pairs]
                avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
                lines.append("")
                lines.append("### QA基准测试词法重叠")
                lines.append("")
                lines.append(f"- 平均词法重叠: {avg_overlap:.3f}")
                lines.append(f"- (参考：启发式标题查询通常重叠 > 0.8)")
                lines.append(f"- (重叠越低 → 词法偏差越小 → 查询越贴近真实用户)")

    return "\n".join(lines)


def generate_report(
    summaries: list[dict],
    qa_path: str | None,
    output_path: str,
) -> None:
    """生成完整的Markdown报告并保存到文件。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_lines = [
        "# RAGAS语义QA评估报告",
        "",
        f"> 生成时间: {now}",
        f"> 评估方法: {', '.join(s['method'] for s in summaries)}",
        "",
        "## 1. 实验配置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
    ]

    # 从第一个摘要添加配置信息
    if summaries:
        s = summaries[0]
        report_lines.extend([
            f"| QA基准测试 | {s.get('qa_path', '—')} |",
            f"| 问题数量 | {s.get('n_successful', '—')} / {s.get('n_questions', '—')} |",
            f"| Top-K | {s.get('top_k', '—')} |",
            f"| 数据集 | {s.get('collection', '—')} |",
        ])

    # 第2节：RAGAS指标对比
    report_lines.extend([
        "",
        "## 2. RAGAS指标对比",
        "",
        generate_method_comparison_table(summaries),
        "",
    ])

    # 第3节：各指标最佳方法
    report_lines.extend([
        "## 3. 各指标最佳方法",
        "",
        generate_best_method_analysis(summaries),
        "",
    ])

    # 第4节：词法偏差分析
    report_lines.extend([
        "## 4. 词法偏差与重排分析",
        "",
        generate_lexical_bias_analysis(summaries, qa_path),
        "",
    ])

    # 第5节：结论
    report_lines.extend([
        "## 5. 结论",
        "",
    ])

    methods = {s["method"] for s in summaries}
    conclusions = []

    if "bm25" in methods and "dense" in methods:
        bm25_s = next(s for s in summaries if s["method"] == "bm25")
        dense_s = next(s for s in summaries if s["method"] == "dense")
        bm25_cr = bm25_s.get("metrics", {}).get("context_recall", {}).get("mean", 0)
        dense_cr = dense_s.get("metrics", {}).get("context_recall", {}).get("mean", 0)

        if bm25_cr > dense_cr:
            conclusions.append(
                "1. BM25即使在语义QA条件下仍保持对Dense检索的优势，"
                "表明其优势并非纯粹源于词法偏差。"
            )
        else:
            conclusions.append(
                "1. Dense检索在语义QA条件下表现优于BM25，"
                "表明启发式基准测试引入了有利于BM25的词法偏差。"
            )

    if "bm25" in methods and "bm25+cross_encoder" in methods:
        conclusions.append(
            "2. CrossEncoder重排在语义QA条件下显示"
            + ("明显" if any(
                s.get("metrics", {}).get("context_precision", {}).get("mean", 0) >
                next(x for x in summaries if x["method"] == "bm25").get("metrics", {}).get("context_precision", {}).get("mean", 0) * 1.05
                for s in summaries if s["method"] == "bm25+cross_encoder"
            ) else "边际")
            + "的上下文精确率改善。"
        )

    conclusions.append(
        "3. 本补充评估使用小规模LLM生成的QA基准测试，"
        "应被解释为方向性证据，而非确定性证明。"
    )

    conclusions.append(
        "4. 为获得更稳健的结论，建议使用人工标注的QA对进行更大规模的评估。"
    )

    report_lines.extend(conclusions)

    # 第6节：局限性
    report_lines.extend([
        "",
        "## 6. 局限性",
        "",
        "- QA对由LLM生成，非人工标注",
        "- 样本量较小（30-50个问题）",
        "- LLM生成质量影响RAGAS评分",
        "- RAGAS指标本身使用LLM评分，引入额外方差",
        "- 结果可能无法泛化到所有查询分布",
    ])

    # 写入报告
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"报告已保存至: {output}")


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从RAGAS结果生成Markdown评估报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                   help="包含summary_*.json文件的目录")
    p.add_argument("--results", nargs="+", default=None,
                   help="显式指定摘要JSON文件列表（覆盖--results_dir）")
    p.add_argument("--qa_path", default="outputs/qa_benchmark/llm_generated_qa.json",
                   help="用于词法重叠分析的QA基准测试路径")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="输出Markdown报告路径")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.results:
        summaries = load_summaries_from_files(args.results)
    else:
        summaries = load_summaries(args.results_dir)

    if not summaries:
        print("[错误] 未找到评估结果。请先运行run_ragas_evaluation.py。")
        sys.exit(1)

    generate_report(
        summaries=summaries,
        qa_path=args.qa_path,
        output_path=args.output,
    )
