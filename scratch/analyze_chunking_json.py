import json

def analyze():
    with open("data/processed/rag_baseline/chunking_experiment/chunking_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("CONTROLLED CHUNKING EXPERIMENT METRICS SUMMARY (K=5):")
    print("="*105)
    print(f"{'Chunking Configuration':<35} | {'Chunks':<6} | {'HitRate':<8} | {'Recall@5':<8} | {'Prec@5':<7} | {'Accuracy':<8} | {'EM':<6} | {'IdxTime':<7}")
    print("-" * 105)

    for item in data["summary_metrics"]:
        print(f"{item['description']:<35} | {item['total_chunks_created']:<6} | {item['retrieval_hit_rate_percent']:<7}% | {item['retrieval_mean_recall_at_5']:<8} | {item['retrieval_mean_precision_at_5']:<7} | {item['correct_rate_percent']:<7}% | {item['exact_match_rate_percent']:<5}% | {item['indexing_time_seconds']:<7}s")

    print("\nFAILURE CATEGORY BREAKDOWN BY CHUNKING STRATEGY:")
    print("="*105)
    categories = [
        "Retrieval Failure",
        "Multi-Hop Reasoning Failure",
        "Generation Failure",
        "Gold-Answer / Data-Quality Issue",
        "Insufficient Context",
        "Answer-Format Mismatch"
    ]
    
    configs = ["small_chunks_100w_20o", "medium_chunks_250w_50o", "large_chunks_500w_100o", "full_document_unchunked"]
    header = f"{'Category':<35} | " + " | ".join([f"{c:<16}" for c in configs])
    print(header)
    print("-" * 105)
    
    for cat in categories:
        counts = [data["results_by_config"][c]["metrics"]["failure_category_counts"].get(cat, 0) for c in configs]
        counts_str = " | ".join([f"{cnt:<16}" for cnt in counts])
        print(f"{cat:<35} | {counts_str}")

if __name__ == "__main__":
    analyze()
