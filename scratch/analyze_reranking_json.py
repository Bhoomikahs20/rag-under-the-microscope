import json

def analyze():
    with open("data/processed/rag_baseline/reranking_experiment/reranking_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    b = data["metrics_summary"]["baseline_vector_top5"]
    r = data["metrics_summary"]["reranked_cross_encoder_top5"]

    print("RERANKING EXPERIMENT METRICS COMPARISON (Top-5 Context):")
    print("="*95)
    print(f"{'Pipeline Setup':<35} | {'HitRate@5':<10} | {'Recall@5':<9} | {'Prec@5':<8} | {'Accuracy':<10} | {'ExactMatch':<10}")
    print("-" * 95)
    print(f"{'Baseline Vector Retrieval (No Rerank)':<35} | {b['hit_rate_percent']:<9}% | {b['recall_at_5']:<9} | {b['precision_at_5']:<8} | {b['answer_accuracy_percent']:<9}% | {b['exact_match_percent']:<9}%")
    print(f"{'Cross-Encoder Reranked (ms-marco)':<35} | {r['hit_rate_percent']:<9}% | {r['recall_at_5']:<9} | {r['precision_at_5']:<8} | {r['answer_accuracy_percent']:<9}% | {r['exact_match_percent']:<9}%")

    print("\nFAILURE CATEGORY BREAKDOWN:")
    print("="*95)
    categories = [
        "Retrieval Failure",
        "Multi-Hop Reasoning Failure",
        "Generation Failure",
        "Gold-Answer / Data-Quality Issue",
        "Insufficient Context",
        "Answer-Format Mismatch"
    ]
    
    header = f"{'Category':<35} | {'Baseline Vector Top-5':<22} | {'Cross-Encoder Reranked Top-5':<25}"
    print(header)
    print("-" * 95)
    
    for cat in categories:
        c_b = b["failure_category_counts"].get(cat, 0)
        c_r = r["failure_category_counts"].get(cat, 0)
        print(f"{cat:<35} | {c_b:<22} | {c_r:<25}")

if __name__ == "__main__":
    analyze()
