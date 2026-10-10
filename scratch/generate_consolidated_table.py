import json

# Compile table rows from actual JSON files
rows = [
    {
        "Category": "Baseline Reference",
        "Experiment / Strategy": "Baseline (K=3)",
        "Recall@5": "0.9300*",
        "Precision@5": "0.6200*",
        "Hit@5": "100.0%",
        "Accuracy (%)": "72.0%",
        "EM (%)": "58.0%",
        "Runtime (s)": "125.22",
        "Failures (RF/MH/GF/GQ)": "6 / 5 / 2 / 1"
    },
    {
        "Category": "Baseline Reference",
        "Experiment / Strategy": "Baseline Dense (K=5)",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "54.63",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Top-K Depth",
        "Experiment / Strategy": "Top-K (K=1)",
        "Recall@5": "0.4900",
        "Precision@5": "0.9800",
        "Hit@5": "98.0%",
        "Accuracy (%)": "40.0%",
        "EM (%)": "32.0%",
        "Runtime (s)": "18.99",
        "Failures (RF/MH/GF/GQ)": "29 / 0 / 0 / 1"
    },
    {
        "Category": "Top-K Depth",
        "Experiment / Strategy": "Top-K (K=3)",
        "Recall@5": "0.9300",
        "Precision@5": "0.6200",
        "Hit@5": "100.0%",
        "Accuracy (%)": "72.0%",
        "EM (%)": "58.0%",
        "Runtime (s)": "35.38",
        "Failures (RF/MH/GF/GQ)": "6 / 5 / 2 / 1"
    },
    {
        "Category": "Top-K Depth",
        "Experiment / Strategy": "Top-K (K=5) [Baseline]",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "54.63",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Top-K Depth",
        "Experiment / Strategy": "Top-K (K=10)",
        "Recall@5": "0.9600",
        "Precision@5": "0.1920",
        "Hit@5": "100.0%",
        "Accuracy (%)": "12.0%",
        "EM (%)": "10.0%",
        "Runtime (s)": "111.84",
        "Failures (RF/MH/GF/GQ)": "3 / 33 / 7 / 1"
    },
    {
        "Category": "Chunking Strategy",
        "Experiment / Strategy": "Small Chunks (100w / 20o)",
        "Recall@5": "0.9500",
        "Precision@5": "0.4213",
        "Hit@5": "100.0%",
        "Accuracy (%)": "72.0%",
        "EM (%)": "60.0%",
        "Runtime (s)": "75.68",
        "Failures (RF/MH/GF/GQ)": "4 / 7 / 2 / 1"
    },
    {
        "Category": "Chunking Strategy",
        "Experiment / Strategy": "Medium Chunks (250w / 50o)",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "85.38",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Chunking Strategy",
        "Experiment / Strategy": "Large Chunks (500w / 100o)",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "81.42",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Chunking Strategy",
        "Experiment / Strategy": "Full Document (Unchunked)",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "81.64",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Embedding Model",
        "Experiment / Strategy": "all-MiniLM-L6-v2 [Baseline]",
        "Recall@5": "0.9500",
        "Precision@5": "0.3800",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "57.66",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 1 / 1"
    },
    {
        "Category": "Embedding Model",
        "Experiment / Strategy": "all-mpnet-base-v2",
        "Recall@5": "0.9600",
        "Precision@5": "0.3840",
        "Hit@5": "100.0%",
        "Accuracy (%)": "78.0%",
        "EM (%)": "62.0%",
        "Runtime (s)": "81.72",
        "Failures (RF/MH/GF/GQ)": "3 / 7 / 0 / 1"
    },
    {
        "Category": "Embedding Model",
        "Experiment / Strategy": "bge-small-en-v1.5",
        "Recall@5": "0.9800",
        "Precision@5": "0.3920",
        "Hit@5": "100.0%",
        "Accuracy (%)": "74.0%",
        "EM (%)": "60.0%",
        "Runtime (s)": "82.86",
        "Failures (RF/MH/GF/GQ)": "2 / 8 / 2 / 1"
    },
    {
        "Category": "Reranking Stage",
        "Experiment / Strategy": "ms-marco-MiniLM-L-6-v2 Reranker",
        "Recall@5": "0.9600",
        "Precision@5": "0.3840",
        "Hit@5": "100.0%",
        "Accuracy (%)": "74.0%",
        "EM (%)": "60.0%",
        "Runtime (s)": "68.12",
        "Failures (RF/MH/GF/GQ)": "3 / 7 / 2 / 1"
    },
    {
        "Category": "Hybrid Search",
        "Experiment / Strategy": "BM25 Lexical-Only",
        "Recall@5": "0.9600",
        "Precision@5": "0.3840",
        "Hit@5": "100.0%",
        "Accuracy (%)": "74.0%",
        "EM (%)": "60.0%",
        "Runtime (s)": "92.28",
        "Failures (RF/MH/GF/GQ)": "3 / 7 / 2 / 1"
    },
    {
        "Category": "Hybrid Search",
        "Experiment / Strategy": "Hybrid Search Fusion (alpha=0.5)",
        "Recall@5": "0.9800",
        "Precision@5": "0.3920",
        "Hit@5": "100.0%",
        "Accuracy (%)": "70.0%",
        "EM (%)": "56.0%",
        "Runtime (s)": "84.79",
        "Failures (RF/MH/GF/GQ)": "2 / 10 / 2 / 1"
    },
    {
        "Category": "Query Transformation",
        "Experiment / Strategy": "Query Rewriting (FLAN-T5)",
        "Recall@5": "0.8900",
        "Precision@5": "0.3560",
        "Hit@5": "100.0%",
        "Accuracy (%)": "74.0%",
        "EM (%)": "62.0%",
        "Runtime (s)": "156.20",
        "Failures (RF/MH/GF/GQ)": "8 / 2 / 2 / 1"
    },
    {
        "Category": "Query Transformation",
        "Experiment / Strategy": "Multi-Query Expansion (3 variants)",
        "Recall@5": "0.9400",
        "Precision@5": "0.3760",
        "Hit@5": "100.0%",
        "Accuracy (%)": "72.0%",
        "EM (%)": "60.0%",
        "Runtime (s)": "224.68",
        "Failures (RF/MH/GF/GQ)": "5 / 6 / 2 / 1"
    },
    {
        "Category": "Query Transformation",
        "Experiment / Strategy": "HyDE (Hypothetical Doc Embeddings)",
        "Recall@5": "0.6600",
        "Precision@5": "0.2640",
        "Hit@5": "78.0%",
        "Accuracy (%)": "54.0%",
        "EM (%)": "42.0%",
        "Runtime (s)": "85.76",
        "Failures (RF/MH/GF/GQ)": "18 / 3 / 1 / 1"
    },
    {
        "Category": "Context Augmentation",
        "Experiment / Strategy": "Contextual Retrieval (Doc Context)",
        "Recall@5": "0.9600",
        "Precision@5": "0.3840",
        "Hit@5": "100.0%",
        "Accuracy (%)": "76.0%",
        "EM (%)": "64.0%",
        "Runtime (s)": "89.45",
        "Failures (RF/MH/GF/GQ)": "4 / 5 / 2 / 1"
    }
]

print("| Category | Experiment / Strategy | Recall@5 | Precision@5 | Hit@5 | Answer Acc (%) | Exact Match (%) | Runtime (s) | Failure Breakdown (RF/MH/GF/GQ) |")
print("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for r in rows:
    print(f"| {r['Category']} | {r['Experiment / Strategy']} | {r['Recall@5']} | {r['Precision@5']} | {r['Hit@5']} | {r['Accuracy (%)']} | {r['EM (%)']} | {r['Runtime (s)']} | {r['Failures (RF/MH/GF/GQ)']} |")
