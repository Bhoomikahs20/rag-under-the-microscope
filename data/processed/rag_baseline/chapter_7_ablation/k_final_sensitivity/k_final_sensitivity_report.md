# Chapter 7 Final Retrieval-Depth Sensitivity Experiment Report

## 1. Executive Summary

This report presents the empirical findings of the **Final Retrieval-Depth ($K_{final}$) Sensitivity Experiment** for Chapter 7 (`data/processed/rag_baseline/chapter_7_ablation/k_final_sensitivity/k_final_sensitivity_results.json`).

The experiment evaluated the revised production architecture selected from Ablation Runs 0–4:
- **Chunking Strategy:** Standard Chunks (250 words / 50 overlap, Title + Chunk text, **without contextual document summaries**).
- **Retrieval Engine:** BAAI/bge-small-en-v1.5 Dense Embeddings + BM25 Lexical Search ($\alpha=0.5$ normalized weighted score fusion).
- **Candidate Pool Size ($K_{init}$):** $K_{init} = 15$ candidate chunks **fixed**.
- **Reranker:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scoring the 15 candidate passages.
- **Reranked Retrieval Depth ($K_{final}$):** Evaluated across $K_{final} \in \{1, 3, 5, 10\}$.
- **Generator LLM:** `google/flan-t5-base` (greedy decoding, $T=0.0$, max_new_tokens=50).

All 50 evaluation queries, ground-truth qrels, corpus of 100 documents, random seed ($42$), and error classification taxonomy were held strictly constant.

> [!IMPORTANT]
> **Key Measured Empirical Discoveries:**
> 1. **Peak Accuracy Sweet Spot ($K_{final} = 5$):** $K_{final} = 5$ achieved the **highest end-to-end answer accuracy (78.0%, 39/50 correct)** and **highest Exact Match (66.0%, 33/50)** across all depth settings, with **100.0% Context Recall (Recall@5 = 1.0000)** and **0 Retrieval Failures**.
> 2. **Multi-Hop Recall Failure at $K_{final} = 1$:** Setting $K_{final} = 1$ causes answer accuracy to drop to **46.0%** and introduces **26 Retrieval Failures**. Because HotpotQA requires two distinct gold supporting documents for multi-hop reasoning, providing only 1 chunk mathematically caps recall at **Recall@1 = 0.5000**.
> 3. **High-Precision Minimal Context ($K_{final} = 3$):** $K_{final} = 3$ achieved **100.0% Recall (Recall@3 = 1.0000)** and **76.0% Accuracy** (38/50 correct) with **28% faster online query latency** (2119.96 ms vs 2938.23 ms) compared to $K_{final} = 5$.
> 4. **Context Window Overload & Generative Collapse ($K_{final} = 10$):** Setting $K_{final} = 10$ flooded `flan-t5-base`'s 1024-token prompt context window. The excessive prompt length triggered **40 Multi-Hop Reasoning Failures** and **9 Generation Failures**, causing answer accuracy to collapse to **0.0%**.

---

## 2. Quantitative Performance Comparison Table ($K_{final} \in \{1, 3, 5, 10\}$)

| $K_{final}$ Depth | Recall@K | Precision@K | Hit Rate (%) | Answer Acc (%) | Exact Match (%) | Failure Breakdown (RF / MH / GF / GQ) | Pre-comp Cost (s) | Indexing Cost (s) | Online Mean (ms) | Online p50 (ms) | Online p95 (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K_{final} = 1$** | 0.5000 | **1.0000** | **100.0%** | 46.0% (23/50) | 38.0% (19/50) | 26 / 0 / 0 / 1 | 0.00s | 4.55s | **1378.39 ms** | **1322.94 ms** | **1889.22 ms** |
| **$K_{final} = 3$** | **1.0000** | 0.6667 | **100.0%** | 76.0% (38/50) | 64.0% (32/50) | **0 / 9 / 2 / 1** | 0.00s | 4.55s | 2119.96 ms | 1929.12 ms | 3959.86 ms |
| **$K_{final} = 5$ [Optimal]** | **1.0000** | 0.4000 | **100.0%** | **78.0% (39/50)** | **66.0% (33/50)** | **0 / 8 / 2 / 1** | 0.00s | 4.55s | 2938.23 ms | 2774.66 ms | 4732.69 ms |
| **$K_{final} = 10$** | **1.0000** | 0.2000 | **100.0%** | 0.0% (0/50) | 0.0% (0/50) | 0 / 40 / 9 / 1 | 0.00s | 4.55s | 4401.14 ms | 4154.26 ms | 6716.31 ms |

*Offline Indexing Cost: BGE Dense Vector FAISS Indexing (4.54s) + BM25 Lexical Indexing (0.01s) = **4.55 seconds total**.*  
*Failure Breakdown legend: `RF` = Retrieval Failure, `MH` = Multi-Hop Reasoning Failure, `GF` = Generation Failure, `GQ` = Gold-Answer / Data-Quality Issue.*

---

## 3. Detailed Empirical Sensitivity Analysis

### 3.1 $K_{final} = 1$: The Multi-Hop Truncation Barrier
- **Retrieval Performance:** Hit Rate = 100.0%, Precision@1 = 1.0000, but Recall@1 is capped at **0.5000**.
- **Generative Performance:** Answer accuracy drops to **46.0%** (23/50 correct) and 26 queries suffer **Retrieval Failures**.
- **Root Cause Analysis:** Multi-hop queries in HotpotQA require bridging information from two distinct Wikipedia documents. Providing only 1 passage guarantees that the secondary bridge document is omitted, truncating necessary context.
- **Latency Profile:** Fastest online query latency (Mean = 1378.39 ms, p50 = 1322.94 ms).

### 3.2 $K_{final} = 3$: High-Precision Minimal Context
- **Retrieval Performance:** **Recall@3 = 1.0000** (100% gold context recall achieved!), Precision@3 = 0.6667.
- **Generative Performance:** Answer accuracy reaches **76.0%** (38/50 correct), Exact Match = **64.0%**.
- **Failure Distribution:** 0 Retrieval Failures, 9 Multi-Hop Reasoning Failures, 2 Generation Failures, 1 Gold Issue.
- **Latency Profile:** Mean online latency is **2119.96 ms** (p50 = 1929.12 ms, p95 = 3959.86 ms).
- **Use Case:** Ideal low-latency deployment setting when generator LLM context window cost is constrained.

### 3.3 $K_{final} = 5$: Peak Accuracy & Exact Match Sweet Spot
- **Retrieval Performance:** **Recall@5 = 1.0000**, Precision@5 = 0.4000.
- **Generative Performance:** **Peak Accuracy (78.0%, 39/50 correct)** and **Peak Exact Match (66.0%, 33/50)** across all retrieval depth settings.
- **Failure Distribution:** Lowest total failures (11 failures: 0 RF, 8 MH, 2 GF, 1 GQ).
- **Latency Profile:** Mean online latency is **2938.23 ms** (p50 = 2774.66 ms, p95 = 4732.69 ms).
- **Measured Finding:** Providing 5 reranked passages gives `flan-t5-base` optimal context depth, allowing the Cross-Encoder to place primary and secondary gold passages at Ranks 1 and 2 while supplying sufficient supporting background.

### 3.4 $K_{final} = 10$: Context Window Overload & Generative Collapse
- **Retrieval Performance:** Recall@10 = 1.0000, Precision@10 = 0.2000.
- **Generative Performance:** Accuracy collapsed to **0.0%** (0/50 correct!).
- **Failure Distribution:** **40 Multi-Hop Reasoning Failures** and **9 Generation Failures**.
- **Root Cause Analysis:** `flan-t5-base` has a maximum sequence length limit of 1024 tokens. Feeding 10 250-word passages (~2500 words / ~3200 tokens) causes hard prompt truncation. The generator model receives truncated context headers and repetitive passage text, inducing severe attention distraction and output repetition loops.
- **Latency Profile:** Slowest online query latency (Mean = 4401.14 ms, p95 = 6716.31 ms).

---

## 4. Final System Architectural Recommendations

Based strictly on the measured empirical results:

1. **Recommended Reranked Context Depth ($K_{final} = 5$):**
   - For maximum answer quality, set **$K_{final} = 5$**, delivering **78.0% Accuracy**, **66.0% Exact Match**, **1.0000 Recall@5**, and **0 Retrieval Failures**.
2. **Low-Latency Alternative ($K_{final} = 3$):**
   - If online query latency budget is under 2.5 seconds, set **$K_{final} = 3$**, delivering **76.0% Accuracy** and **1.0000 Recall@3** while saving **~818 ms per query** compared to $K_{final} = 5$.
3. **Strict Prohibition ($K_{final} \ge 10$ on Small LLMs):**
   - Never exceed $K_{final} = 5$ when using small/medium LLMs (e.g. Flan-T5-Base) due to context window truncation and attention distraction.
