import json

with open("data/processed/rag_baseline/chapter_7_ablation/k_final_sensitivity/k_final_sensitivity_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary_metrics"]

print(f"{'K_final':<8} | {'Recall@K':<9} | {'Prec@K':<8} | {'HitRate':<8} | {'Acc (%)':<7} | {'EM (%)':<6} | {'RF':<3} | {'MH':<3} | {'GF':<3} | {'GQ':<3} | {'Mean Lat (ms)':<14} | {'p50 (ms)':<8} | {'p95 (ms)':<8}")
print("-" * 130)

for k_name, s in summary.items():
    rm = s["retrieval_metrics"]
    gm = s["generation_metrics"]
    fc = s["failure_category_counts"]
    lat = s["online_latency_summary_ms"]["total_online_latency"]
    k_val = s["k_final"]
    
    print(f"{k_val:<8} | {rm['recall_at_k']:<9} | {rm['precision_at_k']:<8} | {rm['hit_rate_percent']:<8} | {gm['answer_accuracy_percent']:<7} | {gm['exact_match_percent']:<6} | {fc['Retrieval Failure']:<3} | {fc['Multi-Hop Reasoning Failure']:<3} | {fc['Generation Failure']:<3} | {fc['Gold-Answer / Data-Quality Issue']:<3} | {lat['mean_ms']:<14} | {lat['p50_ms']:<8} | {lat['p95_ms']:<8}")
