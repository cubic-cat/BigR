# RAGAS 分块策略评估结果

> 评估时间：2026-04-21  
> 脚本：`scripts/eval_strategies.py`

---

## 一、评估配置

| 参数 | 值 |
|------|----|
| 入库文章数（每策略） | 100 篇 |
| 数据来源 | `enwiki_namespace_0_0.jsonl`（前 100 篇非重定向文章） |
| 评估问题数 | 8 个 |
| 检索 top-k | 5 |
| 评估模型（LLM） | `qwen-plus` |
| 评估模型（Embedding） | `text-embedding-v4` |
| RAGAS 版本 | 0.4.3 |

---

## 二、对比策略说明

| 策略名 | 切割方式 | chunk 数（100篇） | 平均 chunk/篇 |
|--------|---------|-----------------|--------------|
| **section** | 以 Wikipedia section 为单位；超过 512 tokens 的 section 二次滑动窗口切割（50 token 重叠） | 195 | 1.95 |
| **fixed512** | 全文拼接后固定 512 token 滑动窗口，50 token 重叠 | 115 | 1.15 |

> `fixed256`（256 token 窗口）已实现但本轮未参与评估，可在下次扩大规模时加入。

---

## 三、评估结果

### 汇总分数（各指标平均值）

```
Metric               section    fixed512
----------------------------------------
context_precision     0.812 *    0.792
context_recall        1.000 *    1.000 *
faithfulness          0.996      1.000 *
answer_relevancy      0.955 *    0.921
----------------------------------------
综合均值             0.941 *    0.928
（* = 该指标最优）
```

**结论：section 策略综合得分更高（0.941 vs 0.928）**

---

### 逐题分数

| 题号 | 问题（简） | cp-sec | cp-512 | cr-sec | cr-512 | fa-sec | fa-512 | ar-sec | ar-512 |
|------|-----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Q1 | 1906飓风 Mobile 最低气压 | 0.33 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q2 | 1214 Richilde 发现时间和人 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 |
| Q3 | NotAgainSU 运动的内容 | 0.75 | **1.00** | 1.00 | 1.00 | 0.97 | 1.00 | **0.89** | 0.86 |
| Q4 | 1214 Richilde 的小行星分类 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.92** | 0.85 |
| Q5 | 新奥尔良风暴潮高度 | **1.00** | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| Q6 | NotAgainSU 提了多少项诉求 | 0.50 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 0.89 | 0.89 |
| Q7 | Tame Impala 2015年的歌 | 0.92 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | **0.97** | 0.78 |
| Q8 | 1214 Richilde 轨道距离 | **1.00** | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

> 列名：`cp`=context_precision，`cr`=context_recall，`fa`=faithfulness，`ar`=answer_relevancy  
> **加粗**= 该题该指标优于另一策略

---

## 四、结果分析

### section 策略的优势

- **Q5、Q8 的 context_precision 明显更高（1.00 vs 0.50）**：fixed512 的固定切割将不相关内容（如其他地区的飓风数据、其他天文数据）混入同一 chunk，降低了检索精准度
- **answer_relevancy 整体更高**：section 的语义完整性使 LLM 生成的回答更聚焦

### fixed512 策略的优势

- **Q3、Q6、Q7 的 context_precision 更高（1.00 vs 0.75/0.50/0.92）**：当一个 section 内容较长且问题只关注其中一小部分时，固定窗口反而能切出更精准的小块
- **faithfulness 略高（1.000 vs 0.996）**：更短的 chunk 减少了 LLM 产生额外幻觉的空间

### 两策略相同的地方

- **context_recall 均为 1.000**：两种策略都能找回所有相关信息，说明 top-k=5 已足够覆盖

---

## 五、指标含义说明

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **context_precision** | 检索回来的 chunks 中，有多少比例是真正相关的（精准率） | 越高越好 |
| **context_recall** | 回答问题所需的信息，有多少比例被检索回来了（召回率） | 越高越好 |
| **faithfulness** | LLM 的回答是否完全基于检索到的上下文（忠实度，越高幻觉越少） | 越高越好 |
| **answer_relevancy** | LLM 的回答是否直接回答了用户的问题（相关性） | 越高越好 |

---

## 六、下一步评估计划

### 建议的下一次评估参数

```powershell
python scripts\eval_strategies.py `
  --max-articles 500 `
  --strategies section,fixed512,fixed256 `
  --top-k 5 `
  --output results\eval_500.json
```

**预计成本：¥1.5，时间：约 50 分钟（入库）+ 30 分钟（评估）**

### 建议补充的评估问题

当前 8 个问题偏向事实类（specific facts），建议补充：
- **多跳问题**：需要跨多个 section 组合信息才能回答
- **摘要类问题**：要求对整篇文章内容概括
- **比较类问题**：涉及同一文章中多个对象的对比

---

## 七、评估用到的 8 个内置问题

```jsonl
{"question": "What was the lowest air pressure recorded during the 1906 Mississippi hurricane?", "reference": "The lowest air pressure recorded in Mobile was 977 mbar during the 1906 Mississippi hurricane."}
{"question": "When was asteroid 1214 Richilde discovered and by whom?", "reference": "Richilde was discovered on 1 January 1932 by German astronomer Max Wolf at the Heidelberg-Königstuhl State Observatory."}
{"question": "What is the #NotAgainSU movement about?", "reference": "NotAgainSU is a hashtag and student-led organization that began after racist incidents at Syracuse University between 2019 and 2021."}
{"question": "What type of asteroid is 1214 Richilde classified as?", "reference": "In the SMASS classification, Richilde is an Xk-subtype asteroid that transitions from X-type to the rare K-type."}
{"question": "What storm surge height was recorded in New Orleans during the 1906 hurricane?", "reference": "A storm surge of about 6 feet (1.8 m) was recorded at the backwater of the Mississippi River in New Orleans."}
{"question": "How many demands did the NotAgainSU protesters make to Syracuse University?", "reference": "The protesters initially made 19 demands to Chancellor Kent Syverud, which was later expanded to 34."}
{"question": "What song by Tame Impala was released in April 2015?", "reference": "'Cause I'm a Man' is a song by Tame Impala released on 7 April 2015 as the second single from Currents."}
{"question": "What is the orbital distance range of asteroid 1214 Richilde from the Sun?", "reference": "Richilde orbits the Sun in the central asteroid belt at a distance of 2.4 to 3.0 AU."}
```
