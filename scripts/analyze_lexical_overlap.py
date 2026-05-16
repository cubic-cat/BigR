#!/usr/bin/env python3
"""计算检索实验测试集的平均词法重叠。

用途
-----
分析不同类型测试集（title/abstract/cross_article/degradation）的词法重叠程度，
评估测试集是否存在有利于BM25的词法偏差。
"""

import json
import re
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def compute_lexical_overlap(query: str, text: str) -> float:
    """计算查询与文本之间的词法重叠率。
    
    使用简单的token重叠：|query_tokens ∩ text_tokens| / |query_tokens|
    只考虑长度≥3的英文单词。
    """
    query_tokens = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
    text_tokens = set(re.findall(r'[a-zA-Z]{3,}', text.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)


def analyze_file(filepath: Path) -> tuple[float, int]:
    """分析单个文件，返回平均词法重叠和样本数。"""
    overlaps = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 根据不同文件格式提取query和source
    filename = filepath.name
    
    if "title" in filename:
        # queries_title.json: {"query_mode": "title", "queries": [...]}
        queries = data.get("queries", data)
        for q in queries:
            query = q["query"]
            title = q.get("title", q.get("source_title", ""))
            overlaps.append(compute_lexical_overlap(query, title))
    
    elif "abstract" in filename:
        # queries_abstract.json: {"query_mode": "abstract", "queries": [...]}
        queries = data.get("queries", data)
        for q in queries:
            query = q["query"]
            title = q.get("title", q.get("source_title", ""))
            overlaps.append(compute_lexical_overlap(query, title))
    
    elif "cross_article" in filename:
        # queries_cross_article.json: {"n_queries": N, "queries": [...]}
        queries = data.get("queries", data)
        for q in queries:
            query = q["query"]
            title = q.get("source_title", "")
            overlaps.append(compute_lexical_overlap(query, title))
    
    elif "degradation" in filename:
        # queries_degradation.json: {"L0:full_sent": [...], "L1:title_removed": [...], ...}
        for level, queries in data.items():
            for q in queries:
                query = q["query"]
                title = q.get("title", q.get("source_title", ""))
                overlaps.append(compute_lexical_overlap(query, title))
    
    if not overlaps:
        return 0.0, 0
    
    avg_overlap = sum(overlaps) / len(overlaps)
    return avg_overlap, len(overlaps)


def main():
    files = [
        "queries_abstract.json",
        "queries_cross_article.json",
        "queries_degradation.json", 
        "queries_title.json",
    ]
    
    print("="*60)
    print("测试集词法重叠分析报告")
    print("="*60)
    
    all_overlaps = []
    
    for filename in files:
        filepath = OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"[警告] 文件不存在: {filename}")
            continue
        
        avg_overlap, n_samples = analyze_file(filepath)
        all_overlaps.extend([avg_overlap] * n_samples)
        
        print(f"\n文件: {filename}")
        print(f"样本数: {n_samples}")
        print(f"平均词法重叠: {avg_overlap:.4f}")
    
    if all_overlaps:
        overall_avg = sum(all_overlaps) / len(all_overlaps)
        print("\n" + "="*60)
        print(f"总体统计")
        print("="*60)
        print(f"总样本数: {len(all_overlaps)}")
        print(f"总体平均词法重叠: {overall_avg:.4f}")
        print("\n分析结论:")
        print(f"  - 词法重叠 > 0.8: 高度词法相关（有利于BM25）")
        print(f"  - 词法重叠 0.4-0.8: 中等词法相关")
        print(f"  - 词法重叠 < 0.4: 低词法相关（更贴近真实用户查询）")


if __name__ == "__main__":
    main()
