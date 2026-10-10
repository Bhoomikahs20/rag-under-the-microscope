import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def examine_shifts():
    with open("data/processed/rag_baseline/reranking_experiment/reranking_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("QUESTIONS WHERE RERANKING MOVED GOLD DOCS INTO TOP-5:")
    print("="*85)
    for q in data["per_question_results"]:
        init_top5_gold = sum(1 for item in q["before_after_rankings"][:5] if item["is_gold_doc"])
        rerank_top5_gold = sum(1 for item in q["before_after_rankings"] if item["is_gold_doc"] and item["reranked_rank"] <= 5)
        
        if rerank_top5_gold > init_top5_gold:
            print(f"Query {q['query_index']} ({q['query_id']}): {q['question']}")
            print(f"  Gold docs in Vector Top-5: {init_top5_gold}/2 -> Reranked Top-5: {rerank_top5_gold}/2")
            for item in q["before_after_rankings"]:
                if item["is_gold_doc"]:
                    print(f"    Gold Doc '{item['title']}' (DocID: {item['doc_id']}): Vector Rank {item['initial_vector_rank']} (Score {item['vector_similarity_score']}) -> Reranked Rank {item['reranked_rank']} (Score {item['cross_encoder_score']})")

    print("\nQUESTIONS WHERE ANSWER ACCURACY CHANGED (BEFORE vs AFTER):")
    print("="*85)
    
    with open("data/processed/rag_baseline/top_k_experiment/top_k_results.json", "r", encoding="utf-8") as f:
        top_k_data = json.load(f)
        
    vec_top5_by_idx = {q["query_index"]: q for q in top_k_data["results_by_k"]["top_5"]["per_question_results"]}
    
    for q in data["per_question_results"]:
        idx = q["query_index"]
        vec_item = vec_top5_by_idx[idx]
        vec_corr = vec_item["correctness_assessment"]["is_correct"]
        rerank_corr = q["correctness_assessment"]["is_correct"]
        
        if vec_corr != rerank_corr:
            status = "IMPROVED (Vector NO -> Reranked YES)" if rerank_corr else "DEGRADED (Vector YES -> Reranked NO)"
            print(f"Query {idx} ({q['query_id']}): {status}")
            print(f"  Q: {q['question']}")
            print(f"  Gold Answer: '{q['gold_answer']}'")
            print(f"  Vector Top-5 Answer: '{vec_item['rag_generated_answer']}'")
            print(f"  Reranked Top-5 Answer: '{q['rag_generated_answer']}'")
            print("-" * 75)

if __name__ == "__main__":
    examine_shifts()
