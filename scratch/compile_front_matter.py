import os
import re

ROOT_DIR = r"d:\Rag book work"
PROCESSED_DIR = r"d:\Rag book work\data\processed\rag_baseline"

# ---------------------------------------------------------
# 1. Front Matter
# ---------------------------------------------------------
FRONT_MATTER = """# Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems
## An Empirical Engineering Guide to Component Evaluation, Infrastructure Architecture, and Adaptive Retrieval

**Authors:** Antigravity AI Engineering & Research Team  
**Publication Date:** October 2026  
**Edition:** First Edition (Audited & Verified Publication Package)  
**Primary Repository:** `d:\\Rag book work`  

---

## Copyright & Disclaimer Notice

**Copyright © 2026 Antigravity AI Engineering Team. All rights reserved.**

Permission is granted to review, cite, and reproduce excerpts of this manuscript for research, educational, and technical evaluation purposes, provided proper credit is given to the source repository.

### Dataset & Software Attribution:
- **Corpus & Queries Dataset:** [`BeIR/hotpotqa`](https://huggingface.co/datasets/BeIR/hotpotqa) (Multi-hop Question Answering Benchmark, licensed under CC BY-SA 4.0).
- **Relevance Annotations:** [`BeIR/hotpotqa-qrels`](https://huggingface.co/datasets/BeIR/hotpotqa-qrels).
- **Underlying Software:** PyTorch, Hugging Face Transformers, Sentence-Transformers, FAISS, PyArrow, rank_bm25.

### Methodological & Empirical Disclaimer:
All numerical benchmark results presented in Chapter 7 and referenced throughout this book are derived from strict, reproducible execution logs under random seed `42` (`data/processed/rag_baseline/final_benchmark/`). 

Proposed future experimental protocols—specifically **Graph RAG** (Chapter 9), **Agentic RAG** (Chapter 10), and **Long-Context LLMs vs. RAG** (Chapter 11)—represent theoretical design specifications and illustrative evaluation frameworks. They are explicitly labeled as **PROPOSED / ILLUSTRATIVE EXPERIMENTS** and were not executed as part of the audited benchmark campaign in Chapter 7. No empirical claims are made regarding their performance.

---

## Preface

The rapid proliferation of Large Language Models (LLMs) has fundamentally transformed software engineering. However, standalone LLMs suffer from two persistent vulnerabilities: knowledge cutoffs and hallucination. Retrieval-Augmented Generation (RAG) emerged as the primary architectural solution, bridging parametric model knowledge with external non-parametric document repositories.

Despite widespread adoption, the RAG ecosystem has become saturated with contradictory claims, complex framework abstractions, and unvalidated techniques. Architectural enhancements such as query rewriting, multi-query expansion, hypothetical document embeddings (HyDE), contextual chunking, and agentic control loops are frequently presented as universal upgrades. Yet, software engineers building production systems face a stark reality: **Which RAG components deliver genuine end-to-end accuracy improvements, which are compute-inefficient, and which actively harm system performance?**

This book was written to answer that question through rigorous empirical measurement. Rather than relying on superficial framework tutorials or synthetic demonstrations, we built a fully transparent, modular RAG research pipeline over the BeIR/HotpotQA benchmark. We evaluated 31 distinct pipeline configurations across 13 experimental phases—culminating in an audited, leakage-safe 40-query held-out test benchmark with empirical percentile bootstrap confidence intervals.

Our findings challenge common assumptions in RAG design. We demonstrate that generator parameter capacity bounds multi-hop reasoning performance, that cross-encoder reranking is the single most cost-effective retrieval upgrade, and that LLM text generation dominates online response latency (88.5% share). Above all, this book establishes a fundamental design principle for modern AI engineering: **Additional RAG complexity must justify its operational cost through measurable performance gains.**

---

## How to Use This Book

This book is structured into four distinct parts, designed for AI engineers, software architects, researchers, and technical leaders:

- **Part I: Foundations & Dataset Schema (Chapters 1–3):** Establishes empirical benchmarking methodology, inspects linked sampling schemas over BeIR/HotpotQA, and builds a clean baseline fixed-pipeline RAG system.
- **Part II: Core Retrieval Components & Exploratory Pilots (Chapters 4–6):** Systematically isolates chunk granularities, retrieval depths, dense embedding models, BM25 lexical search, hybrid fusion, cross-encoder reranking, query transformations, and contextual chunk summaries.
- **Part III: Controlled Ablations & Authoritative Benchmarks (Chapter 7):** Details standardized leave-one-out ablations (Runs 0–4), $K_{\text{final}}$ depth sensitivity, generator model scaling, failure taxonomy audits, and the authoritative held-out test benchmark.
- **Part IV: Architecture, Infrastructure, & Future Paradigms (Chapters 8–12):** Analyzes vector database mechanics, framework abstraction leakage, Knowledge Graph topology (Graph RAG), dynamic control loops (Agentic RAG), production reliability, latency/cost optimization, and emerging frontiers.

---

## Table of Contents

### Part I: Foundations & Dataset Schema
- **Chapter 1: The RAG Dilemma & Empirical Foundations**
  - 1.1 The RAG Promise and the Production Gap
  - 1.2 Why Empirical Benchmarking Matters
  - 1.3 Project Architecture & Execution Pipeline Setup
  - 1.4 Chapter Summary
- **Chapter 2: Dataset Architecture, Schemas, & Linked Sampling**
  - 2.1 The BeIR / HotpotQA Benchmark Dataset
  - 2.2 Linked Sampling via Qrels for 100% Ground Truth Coverage
  - 2.3 Schema Inspection & Manifest Generation
  - 2.4 PyArrow Row-Group Streaming & Performance Optimization
  - 2.5 Chapter Summary
- **Chapter 3: Building the Baseline Fixed-Pipeline RAG System**
  - 3.1 Architecture of the Baseline Pipeline (`PROTO_PILOT_BASELINE`)
  - 3.2 Text Segmentation & Chunking (250w / 50o Standard)
  - 3.3 Dense Vector Embedding & FAISS `IndexFlatIP` Cosine Normalization
  - 3.4 Baseline Generator Setup (`google/flan-t5-base`, 250M)
  - 3.5 Baseline Evaluation Results (72.0% Acc, 58.0% EM) & Failure Analysis
  - 3.6 Chapter Summary

### Part II: Core Retrieval Components & Exploratory Pilots
- **Chapter 4: Granularity Matters: Document Chunking and Retrieval Depth**
  - 4.1 The Role of Chunk Size in Context Precision
  - 4.2 Empirical Chunk Size Evaluation (`PROTO_PILOT_CHUNKING`)
  - 4.3 Retrieval Depth Sensitivity (`PROTO_PILOT_TOP_K`)
  - 4.4 Engineering Trade-offs: Chunk Granularity vs. Generator Attention
  - 4.5 Chapter Summary
- **Chapter 5: Embedding Models, Lexical Search, & Reranking**
  - 5.1 Dense Vector Model Comparison (`PROTO_PILOT_EMBEDDING`)
  - 5.2 Lexical Keyword Retrieval (`BM25Okapi`)
  - 5.3 Hybrid Score Fusion ($\alpha=0.5$ Dense + BM25)
  - 5.4 Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)
  - 5.5 Chapter Summary
- **Chapter 6: Advanced Retrieval Extensions: Query Transformations & Context Augmentation**
  - 6.1 LLM Query Rewriting (`PROTO_PILOT_QUERY_REWRITING`)
  - 6.2 Multi-Query Expansion & RRF (`PROTO_PILOT_MULTI_QUERY`)
  - 6.3 Hypothetical Document Embeddings (`PROTO_PILOT_HYDE`)
  - 6.4 Contextual Chunk Summaries (`PROTO_PILOT_CONTEXTUAL_RETRIEVAL`)
  - 6.5 Negative Results Analysis: Why Advanced Query Extensions Failed
  - 6.6 Chapter Summary

### Part III: Controlled Ablations & Authoritative Benchmarks
- **Chapter 7: What Actually Matters? System-Level Lessons from Controlled Ablations and Benchmark Audits**
  - 7.1 Introduction & The Quest for Empirical RAG Design
  - 7.2 Methodological Rigor, Protocols, & Provenance Architecture
  - 7.3 Master Experimental Comparison Across All Protocols
  - 7.4 Architectural Categorization: Positive, Neutral, and Negative Findings
  - 7.5 Component Ablation Study & $K_{\text{final}}$ Sensitivity Analysis
  - 7.6 Authoritative Final Benchmark Results (`PROTO_FINAL_BENCHMARK`)
  - 7.7 Failure Taxonomy Evolution & Error Breakdown
  - 7.8 Latency Decomposition & Computational Trade-Offs
  - 7.9 "What Actually Matters": Engineering Conclusions & System Limitations

### Part IV: Architecture, Infrastructure, & Future Paradigms
- **Chapter 8: Vector Databases and Frameworks: Architecture, Trade-Offs, Abstraction Leakage, and Operational Realities**
  - 8.1 Introduction & The Infrastructure Layer of RAG
  - 8.2 Fundamentals of Vector Indexing & Similarity Search
  - 8.3 Embedding Lifecycle, Persistence, and Storage Engines
  - 8.4 Metadata Filtering and Hybrid Indexing Mechanics
  - 8.5 Indexing & Real-Time Update Workflows
  - 8.6 Architectural Comparison: Embedded FAISS vs. Production Vector Databases
  - 8.7 Framework Abstractions: Benefits, Hidden Behavior, and Abstraction Leakage
  - 8.8 Scalability, Failure Modes, and Observability Requirements
  - 8.9 Infrastructure Decision Framework & Selection Matrices
  - 8.10 Practical Implementation & Code Repository Guidelines
  - 8.11 Concise Engineering Checklist
  - 8.12 Chapter Summary
- **Chapter 9: Graph RAG: When Relationships Matter More Than Similarity**
  - 9.1 Introduction: Beyond Similarity-Based Retrieval
  - 9.2 Fundamentals of Knowledge Graphs & Graph Retrieval
  - 9.3 Architectural Taxonomy: Vector RAG, Hybrid RAG, and Graph RAG
  - 9.4 Graph RAG Architectures: Dual-Index & Graph-Augmented Pipelines
  - 9.5 Engineering Trade-offs, Failure Modes, & Operational Challenges
  - 9.6 Proposed Illustrative Graph RAG Experiment Protocol (`PROTO_PROPOSED_GRAPH_HYBRID`)
  - 9.7 Decision Framework: When to Use Graph RAG vs. Standard RAG
  - 9.8 Concise Engineering Checklist for Graph RAG
  - 9.9 Chapter Summary
- **Chapter 10: Agentic RAG: Adaptive Retrieval, Planning, and Tool Use**
  - 10.1 Introduction: From Fixed Pipelines to Dynamic Agents
  - 10.2 Core Mechanics of Agentic RAG
  - 10.3 Taxonomy of RAG Paradigms: Fixed to Fully Agentic
  - 10.4 Engineering Failure Modes & Operational Risks
  - 10.5 Proposed Illustrative Agentic RAG Experiment Protocol (`PROTO_PROPOSED_AGENTIC_EVAL`)
  - 10.6 Decision Framework: Fixed Pipeline vs. Agentic RAG
  - 10.7 Observability & Telemetry Requirements for Agentic RAG
  - 10.8 Concise Engineering Checklist for Agentic RAG
  - 10.9 Chapter Summary
- **Chapter 11: Production RAG: Reliability, Evaluation, Cost, and Deployment**
  - 11.1 Introduction: The Transition to Production Engineering
  - 11.2 Production System Architecture & Reliability Engineering
  - 11.3 Latency & Cost Optimization Strategies
  - 11.4 Long-Context LLMs vs. RAG: Comparative Analysis
  - 11.5 Proposed Long-Context vs. RAG Experiment Protocol (`PROTO_PROPOSED_LONGCONTEXT_VS_RAG`)
  - 11.6 Security, Compliance, and Multi-Tenancy
  - 11.7 Deployment Patterns & Continuous Evaluation
  - 11.8 Production Readiness Engineering Checklist
  - 11.9 Chapter Summary
- **Chapter 12: Future of RAG: Adaptive Architectures, Unresolved Frontiers, and the Engineering Discipline**
  - 12.1 Introduction: The Evolution of Retrieval-Augmented Systems
  - 12.2 Empirical Lessons & Unresolved Bottlenecks from Chapters 7–11
  - 12.3 Emerging Architectural Frontiers
  - 12.4 Research Roadmap & Open Engineering Questions
  - 12.5 The Central Engineering Principle: Justifying Complexity Through Measurement
  - 12.6 Conclusion: A Call for Empirical Discipline

---

## List of Figures

- **Figure 1.1:** Conceptual Pipeline Architecture of Retrieval-Augmented Generation Systems (Section 1.1)
- **Figure 2.1:** Linked Sampling Architecture over BeIR/HotpotQA via Row-Group PyArrow Scanning (Section 2.2)
- **Figure 3.1:** Baseline Fixed-Pipeline Execution Flow (`PROTO_PILOT_BASELINE`) (Section 3.1)
- **Figure 4.1:** Chunk Granularity Trade-off: Small vs Medium vs Large Passages (Section 4.1)
- **Figure 5.1:** Cross-Encoder Reranking Architecture over Candidate Pool $K_{\text{init}}=15$ (Section 5.4)
- **Figure 6.1:** Query Transformation Failure Mechanics (Rewriting, Multi-Query, HyDE) (Section 6.5)
- **Figure 7.1:** Architectural Flow of Experimental Protocols and Provenance Hierarchy (Section 7.2)
- **Figure 7.2:** Answer Accuracy and Exact Match Across Standardized Ablation Runs 0–4 (Section 7.5)
- **Figure 7.3:** Component Latency Decomposition (% Share of Total Online Response Time) (Section 7.8)
- **Figure 7.4:** Failure Taxonomy Shift Across Architectural Phases (Section 7.7)
- **Figure 8.1:** Architectural Taxonomy of Exact vs Approximate Vector Search Methods (Section 8.2)
- **Figure 8.2:** The Core Engineering Trade-offs in Vector Retrieval (The ANN Triangle) (Section 8.2)
- **Figure 8.3:** Dual-Storage Paradigm: Decoupling Vector Search from Payload Hydration (Section 8.3)
- **Figure 8.4:** Comparison of Metadata Filtering Mechanics in Vector Databases (Section 8.4)
- **Figure 8.5:** Abstraction Leakage: High-Level Framework Chains Hiding Subsystem Behavior (Section 8.7)
- **Figure 8.6:** Primary Failure Modes in Production Vector Retrieval Pipeline Infrastructure (Section 8.8)
- **Figure 8.7:** Decision Tree for Vector Infrastructure Architecture (Section 8.9)
- **Figure 8.8:** Recommended Software Design Pattern for Vector Store Decoupling (Section 8.10)
- **Figure 9.1:** Vector Similarity Proximity vs Explicit Knowledge Graph Topology (Section 9.1)
- **Figure 9.2:** The Pipeline Architecture for Knowledge Graph Construction from Unstructured Text (Section 9.2)
- **Figure 9.3:** The Step-by-Step Subgraph Traversal and Context Linearization Workflow (Section 9.2)
- **Figure 9.4:** Dual-Index Architectural Pattern Integrating Vector Retrieval with Knowledge Graph Expansion (Section 9.4)
- **Figure 9.5:** Taxonomy of Infrastructure and Traversal Failures in Graph RAG Systems (Section 9.5)
- **Figure 9.6:** Pragmatic Decision Framework for Evaluating Graph RAG Adoption (Section 9.7)
- **Figure 10.1:** Linear Fixed-Pipeline Control Flow vs. Dynamic Agentic Control Loops (Section 10.1)
- **Figure 10.2:** Detailed Architecture of the Adaptive Control Loop, Verification Cycle, and Tool Engine (Section 10.2)
- **Figure 10.3:** Primary Infrastructure Failure Modes in Agentic RAG Systems (Section 10.4)
- **Figure 10.4:** Pragmatic Decision Framework for Evaluating Agentic RAG Deployment (Section 10.6)
- **Figure 10.5:** Telemetry Architecture for Logging Agentic Execution Trajectories (Section 10.7)
- **Figure 11.1:** Production-Grade RAG Architecture with Caching, Decoupled Retrieval, and Accelerated Generation (Section 11.2)
- **Figure 11.2:** Four-Stage Strategy for Reducing Response Latency and Cloud Compute Expenditure (Section 11.3)
- **Figure 11.3:** Architectural Comparison: Ingesting Massive Context Windows vs. Targeted Vector Retrieval (Section 11.4)
- **Figure 11.4:** Multi-Layer Security Architecture: Input Sanitization, ACL Bitmask Filtering, and PII Masking (Section 11.6)
- **Figure 11.5:** The RAG Triad Evaluation Framework for Continuous Telemetry Monitoring (Section 11.7)
- **Figure 12.1:** The Four Generational Phases of Retrieval-Augmented Generation (Section 12.1)
- **Figure 12.2:** Key Unresolved Engineering Bottlenecks Identified in Chapters 7--11 (Section 12.2)
- **Figure 12.3:** Strategic Research Roadmap for Next-Generation Retrieval Systems (Section 12.4)

---

## List of Tables

- **Table 1.1:** Summary Overview of Experimental Phases and Research Objectives (Section 1.3)
- **Table 2.1:** BeIR / HotpotQA Dataset Subsets and Field Definitions (Section 2.1)
- **Table 2.2:** Pilot Corpus Manifest Statistics and Linked Relevance Summary (Section 2.3)
- **Table 3.1:** Baseline Fixed-Pipeline Configuration Specifications (`PROTO_PILOT_BASELINE`) (Section 3.1)
- **Table 3.2:** Baseline Benchmark Evaluation Metrics and Failure Mode Distribution (Section 3.5)
- **Table 4.1:** Chunk Size Evaluation Results across Small, Medium, and Large Granularities (Section 4.2)
- **Table 4.2:** Top-K Retrieval Depth Sensitivity Analysis ($K \in \{1, 3, 5, 10\}$) (Section 4.3)
- **Table 5.1:** Embedding Model Performance Comparison (`MiniLM`, `BGE-Small`, `E5-Small`) (Section 5.1)
- **Table 5.2:** Dense vs. Lexical BM25 vs. Hybrid Fusion Performance Matrix (Section 5.3)
- **Table 5.3:** Cross-Encoder Reranking Retrieval and Accuracy Enhancement (Section 5.4)
- **Table 6.1:** Performance Comparison of Advanced Query Extensions vs Raw Baseline (Section 6.5)
- **Table 7.1:** Master RAG Architectural Benchmark Comparison across All Protocols (Section 7.3)
- **Table 7.2:** Component Impact Categorization (Positive, Neutral, Negative) (Section 7.4)
- **Table 7.3:** Standardized Component Ablation Runs 0–4 (`PROTO_STD_CHAPTER_7_ABLATION`) (Section 7.5)
- **Table 7.4:** Retrieval Depth ($K_{\text{final}}$) Sensitivity Analysis (`PROTO_STD_CHAPTER_7_KFINAL`) (Section 7.5)
- **Table 7.5:** Authoritative Final Benchmark Results with Bootstrap 95% CIs (`PROTO_FINAL_BENCHMARK`) (Section 7.6)
- **Table 7.6:** Component-Level Online Latency Decomposition (Test Set N=40) (Section 7.8)
- **Table 8.1:** Master Architectural Comparison: Embedded Vector Libraries vs. Production Vector Databases (Section 8.6)
- **Table 9.1:** Comprehensive Architectural Comparison Across Retrieval Paradigms (Section 9.3)
- **Table 9.2:** Pragmatic Selection Framework: Standard RAG vs Hybrid RAG vs Graph RAG (Section 9.7)
- **Table 10.1:** Architectural Comparison Across RAG Paradigms (Fixed to Fully Agentic) (Section 10.3)
- **Table 10.2:** Pragmatic Infrastructure Selection Matrix (Fixed vs Planner vs Agentic) (Section 10.6)
- **Table 11.1:** Production Reliability & Fallback Matrix (Section 11.2)
- **Table 11.2:** Technical Trade-off Matrix: Long-Context LLMs vs. Production RAG (Section 11.4)
- **Table 12.1:** Architectural Progression vs. Operational Complexity Matrix (Section 12.5)

---
"""

print("Front Matter generated successfully.")
