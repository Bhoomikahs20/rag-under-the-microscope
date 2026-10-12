# RAG Book: Retrieval-Augmented Generation — From Baseline Pipelines to Production Systems

Welcome to the official repository for **"Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems (An Empirical Engineering Guide to Component Evaluation, Infrastructure Architecture, and Adaptive Retrieval)"** by **Bhoomika H S**.

---

## 📌 Executive Summary & Key Contributions

This project provides an empirically audited, end-to-end research and engineering guide to Retrieval-Augmented Generation (RAG). Built around the multi-hop question answering benchmark **HotpotQA** (`BeIR/hotpotqa`), this codebase evaluates 31 distinct pipeline configurations across 13 experimental phases—culminating in a leakage-safe held-out benchmark split with empirical 95% bootstrap confidence intervals.

### Primary Engineering Takeaways:
1. **Generator Capacity Bounds Performance:** Perfect retrieval recall (Recall@5 = 1.0000) is insufficient if the generator model lacks parametric reasoning capacity. Generator model scaling yielded the single largest accuracy gain in our campaign (+12.0% gain over baseline).
2. **Cross-Encoder Reranking Delivers High Precision:** Reranking over a hybrid candidate pool ($K_{\text{init}}=15$) drove top-5 retrieval recall to **100.0%** while adding less than 800 ms of online latency.
3. **Query Transformations Can Degrade Accuracy:** Small LLM query rewriters, multi-query expansion pipelines, and hypothetical document generation (HyDE) introduce semantic drift and hallucinations, reducing end-to-end answer accuracy by 4.0% to 10.0%.
4. **Contextual Document Summaries Are Cost-Inefficient:** Prepending LLM-generated document summaries to text chunks added **1,220 seconds** of offline precomputation time without improving answer accuracy (0.0% gain, -2.0% Exact Match regression due to token dilution).
5. **Generation Dominates System Latency:** In a production RAG pipeline, LLM text generation accounts for **88.5% of total online response latency** (6,436.84 ms out of 7,272.35 ms mean), dwarfing dense retrieval (0.6%) and cross-encoder reranking (10.8%).

---

## 📊 Authoritative Final Benchmark Results (`PROTO_FINAL_BENCHMARK`)

Evaluated on the held-out **40-query Test Set** ($N=40$) under frozen hyperparameters (seed `42`, greedy decoding $T=0.0$):

| Performance Metric | Point Estimate ($N=40$ Test) | Empirical 95% Bootstrap Confidence Interval | Dev Set Result ($N=10$) |
| :--- | :---: | :---: | :---: |
| **Answer Accuracy (%)** | **90.00%** (36/40) | **[80.00%, 97.50%]** | 100.00% (10/10) |
| **Exact Match Rate (%)** | **85.00%** (34/40) | **[72.50%, 95.00%]** | 80.00% (8/10) |
| **Hit Rate @ 5 (%)** | **100.00%** (40/40) | **[100.00%, 100.00%]** | 100.00% (10/10) |
| **Recall @ 5** | **1.0000** | **[1.0000, 1.0000]** | 1.0000 |
| **Precision @ 5** | **0.4000** | **[0.4000, 0.4000]** | 0.4000 |
| **Total Online Response Latency**| **7,272.35 ms** (7.27 s) | **[6,768.42 ms, 7,874.62 ms]** | 9,541.95 ms |
| **LLM Generation Latency Share**| **6,436.84 ms** (88.5%) | **[5,959.18 ms, 7,021.75 ms]** | - |
| **Total Offline Indexing Time** | **13.3266 s** | - | - |

*Failure Breakdown:* 0 Retrieval Failures (`RF`), 3 Multi-Hop Reasoning Failures (`MH`), 0 Generation Failures (`GF`), 0 Insufficient Context (`IC`), 1 Data-Quality Issue (`DQ` on QID `5adf48b25542995ec70e8fb5`).

---

## 📁 Repository & Code Structure

```text
d:\Rag book work├── README.md                   # Main project overview & benchmark documentation
├── requirements.txt            # Python dependencies
├── config/
│   └── config.yaml             # Dataset HF paths, random seed 42, and baseline parameters
├── data/
│   ├── processed/
│   │   ├── hotpotqa_pilot_queries.json
│   │   ├── hotpotqa_pilot_corpus.json
│   │   ├── hotpotqa_pilot_qrels.json
│   │   ├── chapter_7_ablation/         # Master results JSON & ablation logs
│   │   ├── final_benchmark/           # Authoritative benchmark JSON, logs, & CIs
│   │   ├── publication/               # Publication package (DOCX, PDF)
│   │   ├── full_book_final_audit.md   # Complete manuscript audit report
│   │   ├── rag_book_publication_qa.md # Publication QA checklist
│   │   └── rag_book_final_manuscript.md # Assembled manuscript source
│   └── manifest.json           # Reproducible dataset schema manifest
├── src/
│   ├── data/
│   │   ├── loader.py           # PyArrow streaming loader & schema inspection
│   │   └── manifest.py         # Manifest validator
│   ├── rag/
│   │   ├── chunker.py          # Sliding window text chunking (250w/50o)
│   │   ├── retriever.py        # FAISS dense retriever & evaluation metrics
│   │   └── generator.py        # FLAN-T5 generator wrapper (greedy decoding)
│   └── utils/
│       └── logging_config.py   # Microsecond timing and logger setup
└── scripts/                    # Reproducible experiment execution scripts (Phases 1--13)
```

---

## 🚀 How to Reproduce Benchmark Results

### 1. Environment Setup
```powershell
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install python-docx reportlab markdown
```

### 2. Stream Dataset & Execute Pipelines
```powershell
# Stream BeIR/hotpotqa dataset via PyArrow & build manifest
.\.venv\Scripts\python.exe scripts/load_pilot_dataset.py

# Execute Chapter 7 Ablation Campaign (Runs 0--4)
.\.venv\Scripts\python.exe scripts/run_hybrid_experiment.py
```

### 3. Generate Manuscript & Publication Package
```powershell
# Assemble final manuscript and export DOCX & PDF packages
.\.venv\Scripts\python.exe scratch/build_full_manuscript.py
.\.venv\Scripts\python.exe scratch/generate_publication_package.py
```

---

## 📖 Book Structure (Chapters 1–12)

- **Part I: Foundations & Dataset Schema**
  - Chapter 1: The RAG Dilemma & Empirical Foundations
  - Chapter 2: Dataset Architecture, Schemas, & Linked Sampling
  - Chapter 3: Building the Baseline Fixed-Pipeline RAG System
- **Part II: Core Retrieval Components & Exploratory Pilots**
  - Chapter 4: Granularity Matters: Document Chunking and Retrieval Depth
  - Chapter 5: Embedding Models, Lexical Search, & Reranking
  - Chapter 6: Advanced Retrieval Extensions: Query Transformations & Context Augmentation
- **Part III: Controlled Ablations & Authoritative Benchmarks**
  - Chapter 7: What Actually Matters? System-Level Lessons from Controlled Ablations and Benchmark Audits
- **Part IV: Architecture, Infrastructure, & Future Paradigms**
  - Chapter 8: Vector Databases and Frameworks: Architecture, Trade-Offs, Abstraction Leakage, and Operational Realities
  - Chapter 9: Graph RAG: When Relationships Matter More Than Similarity
  - Chapter 10: Agentic RAG: Adaptive Retrieval, Planning, and Tool Use
  - Chapter 11: Production RAG: Reliability, Evaluation, Cost, and Deployment
  - Chapter 12: Future of RAG: Adaptive Architectures, Unresolved Frontiers, and the Engineering Discipline

---

## 📄 Citation Information

If you utilize this repository, evaluation framework, or empirical findings in your research or production systems, please cite as:

```bibtex
@book{hs2026rag,
  title     = {Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems},
  subtitle  = {An Empirical Engineering Guide to Component Evaluation, Infrastructure Architecture, and Adaptive Retrieval},
  author    = {H S, Bhoomika},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23276026},
  url       = {https://zenodo.org/records/23276026}
}
```
