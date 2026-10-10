import json

def analyze():
    with open("data/processed/rag_baseline/embedding_experiment/embedding_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("EMBEDDING MODEL COMPARISON METRICS SUMMARY (K=5):")
    print("="*95)
    print(f"{'Embedding Model':<40} | {'HitRate':<8} | {'Recall@5':<8} | {'Prec@5':<7} | {'Accuracy':<8} | {'EM':<6} | {'IdxTime':<7} | {'TotalTime':<9}")
    print("-" * 95)

    for item in data["summary_metrics"]:
        print(f"{item['embedding_model']:<40} | {item['retrieval_hit_rate_percent']:<7}% | {item['retrieval_mean_recall_at_5']:<8} | {item['retrieval_mean_precision_at_5']:<7} | {item['correct_rate_percent']:<7}% | {item['exact_match_rate_percent']:<5}% | {item['indexing_time_seconds']:<7}s | {item['evaluation_runtime_seconds']:<9}s")

    print("\nFAILURE CATEGORY BREAKDOWN BY MODEL:")
    print("="*95)
    categories = [
        "Retrieval Failure",
        "Multi-Hop Reasoning Failure",
        "Generation Failure",
        "Gold-Answer / Data-Quality Issue",
        "Insufficient Context",
        "Answer-Format Mismatch"
    ]
    
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "bge-small-en-v1.5"]
    header = f"{'Category':<35} | " + " | ".join([f"{m:<18}" for m in models])
    print(header)
    print("-" * 95)
    
    for cat in categories:
        counts = [data["results_by_model"][m]["metrics"]["failure_category_counts"].get(cat, 0) for m in models]
        counts_str = " | ".join([f"{c:<18}" for c in counts])
        print(f"{cat:<35} | {counts_str}")

if __name__ == "__main__":
    analyze()
