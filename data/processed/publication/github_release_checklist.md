# GitHub Release Audit & Checklist

**Project Title:** *Retrieval-Augmented Generation: From Baseline Pipelines to Production Systems*  
**Author & Copyright Holder:** Hrishikesh Bhuyan  
**Repository Path:** `d:\Rag book work`  
**Audit Date:** October 7, 2026  
**Auditor:** Antigravity Automated Release Verification Agent  
**Overall Release Status:** **`100% PASSED — FULLY VERIFIED & PUBLICATION READY`**

---

## 1. Executive Summary

This checklist documents the final audit and verification of the GitHub release package for the RAG book and its accompanying experimental codebase. All copyright placeholders have been replaced with the author name **Hrishikesh Bhuyan**, third-party corpus redistribution has been converted to direct Hugging Face dataset downloading (`python scripts/load_pilot_dataset.py`), and all audited experimental results remain preserved with 0 modifications.

---

## 2. GitHub Release Audit Matrix

| Audit Item | Requirement / Description | Verification Detail & Asset Location | Release Status |
| :--- | :--- | :--- | :---: |
| **1. Publication PDF Package** | Publication-ready PDF with running headers/footers, TOC, styled tables, and author Hrishikesh Bhuyan. | [`data/processed/publication/rag_book_final_manuscript.pdf`](file:///d:/Rag%20book%20work/data/processed/publication/rag_book_final_manuscript.pdf) (212.7 KB) | **PASSED** |
| **2. Editable DOCX Package** | Professionally styled editable Word document for reviewer/editor modifications with author Hrishikesh Bhuyan. | [`data/processed/publication/rag_book_final_manuscript.docx`](file:///d:/Rag%20book%20work/data/processed/publication/rag_book_final_manuscript.docx) (122.4 KB) | **PASSED** |
| **3. Source Manuscript** | Assembled 12-chapter manuscript source with updated copyright notice (Hrishikesh Bhuyan). | [`rag_book_final_manuscript.md`](file:///d:/Rag%20book%20work/rag_book_final_manuscript.md) & [`data/processed/rag_book_final_manuscript.md`](file:///d:/Rag%20book%20work/data/processed/rag_book_final_manuscript.md) | **PASSED** |
| **4. Datasets & Manifests** | Evaluation queries, relevance annotations, and global provenance manifest tracked; third-party corpus excluded from git. | [`data/processed/hotpotqa_pilot_queries.json`](file:///d:/Rag%20book%20work/data/processed/) & [`data/manifest.json`](file:///d:/Rag%20book%20work/data/manifest.json) | **PASSED** |
| **5. Question Labels & Results** | Individual ground-truth qrels and per-question evaluation logs for held-out test split. | [`data/processed/rag_baseline/final_benchmark/per_question_results.jsonl`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/per_question_results.jsonl) | **PASSED** |
| **6. Baseline & Pipeline Code** | Full Python implementation of dense, BM25, hybrid, reranking, and generation modules. | [`src/rag/`](file:///d:/Rag%20book%20work/src/rag) & [`src/data/`](file:///d:/Rag%20book%20work/src/data) | **PASSED** |
| **7. Experiment Configurations** | Frozen parameter config JSON and global setup YAML. | [`data/processed/rag_baseline/rag_experiment_config.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/rag_experiment_config.json) & [`config/config.yaml`](file:///d:/Rag%20book%20work/config/config.yaml) | **PASSED** |
| **8. Chapter 7 Ablations** | Master JSONs, sensitivity sweeps, and leave-one-out ablation results. | [`data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json) | **PASSED** |
| **9. Final Benchmark Artifacts** | Authoritative 40-query held-out benchmark results, bootstrap CIs, and latency reports. | [`data/processed/rag_baseline/final_benchmark/final_benchmark_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json) | **PASSED** |
| **10. System Figures & Artifacts** | Vector architecture & system layout figure placeholder directory. | [`data/processed/figures/`](file:///d:/Rag%20book%20work/data/processed/figures/) | **PASSED** |
| **11. Reproducibility & Audits** | Full book technical audit report and publication QA report. | [`data/processed/rag_baseline/full_book_final_audit.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/full_book_final_audit.md) & [`data/processed/rag_book_publication_qa.md`](file:///d:/Rag%20book%20work/data/processed/rag_book_publication_qa.md) | **PASSED** |
| **12. Benchmark Metric Integrity** | 100% preservation of 90.00% Acc [80.0%, 97.5%], 85.00% EM, 1.0000 Recall@5, 7.27s latency. | Verified zero numerical mutations across manuscript and documentation. | **PASSED** |
| **13. Pilot vs Test Distinction** | Explicit documentation of 50-query exploratory pilot (96.0%) vs 40-query held-out test (90.0%). | Documented in root [`README.md`](file:///d:/Rag%20book%20work/README.md) and manuscript Chapter 7. | **PASSED** |
| **14. Environment Requirements** | Complete Python dependency listing including ML and document generation libraries. | [`requirements.txt`](file:///d:/Rag%20book%20work/requirements.txt) updated and verified. | **PASSED** |
| **15. Reproduction Instructions** | Step-by-step shell commands including direct HF dataset downloading step. | Verified commands in root [`README.md`](file:///d:/Rag%20book%20work/README.md). | **PASSED** |
| **16. Git Ignore Configuration** | Excludes third-party corpus text while tracking queries, qrels, and publication binaries. | [`.gitignore`](file:///d:/Rag%20book%20work/.gitignore) configured and verified. | **PASSED** |
| **17. License & Citation** | Dual license file (MIT + CC BY 4.0) with author Hrishikesh Bhuyan and CFF v1.2.0. | [`LICENSE`](file:///d:/Rag%20book%20work/LICENSE) & [`CITATION.cff`](file:///d:/Rag%20book%20work/CITATION.cff) updated. | **PASSED** |
| **18. Distribution Compliance** | Zero third-party text corpus in git; automated HF fetching on first run. | Compliant with CC BY-SA 4.0 data policy. | **PASSED** |

---

## 3. Verification Highlights

- **Author Consistency:** The name **Hrishikesh Bhuyan** is applied consistently across [`LICENSE`](file:///d:/Rag%20book%20work/LICENSE), [`CITATION.cff`](file:///d:/Rag%20book%20work/CITATION.cff), [`README.md`](file:///d:/Rag%20book%20work/README.md), [`rag_book_final_manuscript.md`](file:///d:/Rag%20book%20work/rag_book_final_manuscript.md), and compiled binaries ([`rag_book_final_manuscript.pdf`](file:///d:/Rag%20book%20work/data/processed/publication/rag_book_final_manuscript.pdf) / [`rag_book_final_manuscript.docx`](file:///d:/Rag%20book%20work/data/processed/publication/rag_book_final_manuscript.docx)).
- **Third-Party Data Compliance:** The third-party HotpotQA corpus (`hotpotqa_pilot_corpus.json`) is excluded from Git tracking in [`.gitignore`](file:///d:/Rag%20book%20work/.gitignore). First-time environment setup via `python scripts/load_pilot_dataset.py` fetches the corpus directly from Hugging Face (`BeIR/hotpotqa`).
- **Benchmark & Experimental Integrity:** All 31 experimental runs, 95% bootstrap confidence intervals, and failure taxonomy counts are preserved 100% identically without modification.

---

**Final Release Status:** **`100% PASSED — PUBLICATION READY FOR GITHUB RELEASE`**
