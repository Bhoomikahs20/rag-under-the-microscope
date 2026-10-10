import json

def analyze():
    with open("data/processed/rag_baseline/hybrid_search_experiment/hybrid_search_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("HYBRID SEARCH EXPERIMENT METRICS SUMMARY (K=5):")
    print("="*95)
    print(f"{'Retrieval Mode':<30} | {'HitRate':<8} | {'Recall@5':<8} | {'Prec@5':<7} | {'Accuracy':<8} | {'EM':<6} | {'Time(s)':<8}")
    print("-" * 95)

    for item in data["summary_metrics"]:
        print(f"{item['retrieval_mode']:<30} | {item['retrieval_hit_rate_percent']:<7}% | {item['retrieval_mean_recall_at_5']:<8} | {item['retrieval_mean_precision_at_5']:<7} | {item['correct_rate_percent']:<7}% | {item['exact_match_rate_percent']:<5}% | {item['pipeline_runtime_seconds']:<8}s")

    print("\nFAILURE CATEGORY BREAKDOWN BY MODE:")
    print("="*95)
    categories = [
        "Retrieval Failure",
        "Multi-Hop Reasoning Failure",
        "Generation Failure",
        "Gold-Answer / Data-Quality Issue",
        "Insufficient Context",
        "Answer-Format Mismatch"
    ]
    
    modes = ["dense_only", "bm25_only", "hybrid_fusion_alpha_0.5"]
    header = f"{'Category':<35} | " + " | ".join([f"{m:<20}" for m in modes])
    print(header)
    print("-" * 95)
    
    for cat in categories:
        counts = [data["results_by_mode"][m]["metrics"]["failure_category_counts"].get(cat, 0) for m in modes]
        counts_str = " | ".join([f"{c:<20}" for c in counts])
        print(f"{cat:<35} | {counts_str}")

if __name__ == "__main__":
    analyze()
