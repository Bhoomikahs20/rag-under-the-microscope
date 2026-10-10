import json

with open("data/processed/rag_baseline/chapter_7_ablation/chapter_7_ablation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary_metrics"]

print(f"{'Run ID':<8} | {'Recall@5':<8} | {'Prec@5':<7} | {'Hit@5':<6} | {'Acc (%)':<7} | {'EM (%)':<6} | {'RF':<3} | {'MH':<3} | {'GF':<3} | {'GQ':<3} | {'Precomp (s)':<11} | {'Index (s)':<9} | {'Mean Online (ms)':<16} | {'p50 (ms)':<8} | {'p95 (ms)':<8}")
print("-" * 140)

for run_id, s in summary.items():
    rm = s["retrieval_metrics"]
    gm = s["generation_metrics"]
    fc = s["failure_category_counts"]
    oc = s["offline_cost"]
    lat = s["online_latency_summary_ms"]["total_online_latency"]
    
    print(f"{run_id:<8} | {rm['recall_at_5']:<8} | {rm['precision_at_5']:<7} | {rm['hit_rate_percent']:<6} | {gm['answer_accuracy_percent']:<7} | {gm['exact_match_percent']:<6} | {fc['Retrieval Failure']:<3} | {fc['Multi-Hop Reasoning Failure']:<3} | {fc['Generation Failure']:<3} | {fc['Gold-Answer / Data-Quality Issue']:<3} | {oc['context_summarization_precomputation_seconds']:<11} | {oc['index_creation_seconds']:<9} | {lat['mean_ms']:<16} | {lat['p50_ms']:<8} | {lat['p95_ms']:<8}")
