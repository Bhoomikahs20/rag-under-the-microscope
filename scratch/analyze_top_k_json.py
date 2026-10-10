import json

def analyze():
    with open("data/processed/rag_baseline/top_k_experiment/top_k_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("TOP-K EXPERIMENT METRICS SUMMARY:")
    print("="*85)
    print(f"{'K':<4} | {'HitRate@K':<10} | {'Recall@K':<9} | {'Prec@K':<8} | {'Accuracy':<10} | {'ExactMatch':<10} | {'Time(s)':<8}")
    print("-" * 85)

    for item in data["k_summary"]:
        print(f"{item['top_k']:<4} | {item['retrieval_hit_rate_percent']:<9}% | {item['retrieval_mean_recall_at_k']:<9} | {item['retrieval_mean_precision_at_k']:<8} | {item['correct_rate_percent']:<9}% | {item['exact_match_rate_percent']:<9}% | {item['runtime_seconds']:<8}")

    print("\nFAILURE CATEGORY BREAKDOWN BY K:")
    print("="*85)
    categories = [
        "Retrieval Failure",
        "Multi-Hop Reasoning Failure",
        "Generation Failure",
        "Gold-Answer / Data-Quality Issue",
        "Insufficient Context",
        "Answer-Format Mismatch"
    ]
    
    header = f"{'Category':<35} | " + " | ".join([f"K={k:<2}" for k in [1, 3, 5, 10]])
    print(header)
    print("-" * 85)
    
    for cat in categories:
        counts = [data["results_by_k"][f"top_{k}"]["metrics"]["failure_category_counts"].get(cat, 0) for k in [1, 3, 5, 10]]
        counts_str = " | ".join([f"{c:<4}" for c in counts])
        print(f"{cat:<35} | {counts_str}")

if __name__ == "__main__":
    analyze()
