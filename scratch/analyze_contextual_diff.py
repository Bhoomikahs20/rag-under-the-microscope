import json

with open("data/processed/rag_baseline/contextual_retrieval_experiment/contextual_retrieval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("--- RECALL & RETRIEVAL CHANGES ---")
for r in data["per_question_results"]:
    qid = r["query_id"]
    idx = r["query_index"]
    orig_gold = r["all_gold_retrieved_original"]
    ctx_gold = r["all_gold_retrieved_contextual"]
    is_corr = r["correctness_assessment"]["is_correct"]
    fail_cat = r["failure_category"]
    
    if orig_gold != ctx_gold or fail_cat is not None:
        print(f"Q{idx:02d} ({qid}): orig_all_gold={orig_gold}, ctx_all_gold={ctx_gold} | correct={is_corr} | failure={fail_cat}")
        print(f"   Question: {r['question']}")
        print(f"   RAG Answer: {r['rag_generated_answer']} | Gold: {r['gold_answer']}")
        print()
