"""
Closed-Book Evaluation for 50 Linked HotpotQA Pilot Queries
"""

import os
import sys
import json
import re
import string
import time
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Ensure reproducible torch & python execution
torch.manual_seed(42)

def normalize_answer(s: str) -> str:
    """Lowercases text, removes punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_correctness(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Computes exact match, F1 score, and binary correctness assessment."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    # Exact Match
    exact_match = (norm_pred == norm_gold)

    # Substring / Containment Match
    substring_match = (norm_gold in norm_pred) or (norm_pred in norm_gold and len(norm_pred) > 2)

    # Token F1
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        f1 = float(exact_match)
    elif num_same == 0:
        f1 = 0.0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

    # Correctness decision: Either exact match or high F1 / substring match
    is_correct = exact_match or (f1 >= 0.8) or (substring_match and len(norm_pred) > 0)

    return {
        "exact_match": exact_match,
        "f1_score": round(f1, 4),
        "substring_match": substring_match,
        "is_correct": bool(is_correct)
    }


def run_closed_book_eval():
    output_dir = "data/processed/closed_book"
    os.makedirs(output_dir, exist_ok=True)

    queries_file = "data/processed/hotpotqa_pilot_queries.json"
    if not os.path.exists(queries_file):
        print(f"Error: Pilot queries file not found at {queries_file}")
        return

    with open(queries_file, "r", encoding="utf-8") as f:
        pilot_queries = json.load(f)

    pilot_qids = set(q["_id"] for q in pilot_queries)
    print(f"Loaded {len(pilot_queries)} pilot queries.")

    # 1. Load Gold Answers from HotpotQA dataset
    print("Fetching gold answers from HuggingFace hotpotqa/hotpot_qa dataset...")
    hp_ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train", streaming=True)
    gold_answers = {}
    
    for item in hp_ds:
        qid = item["id"]
        if qid in pilot_qids:
            gold_answers[qid] = item["answer"]
            if len(gold_answers) == len(pilot_qids):
                break

    print(f"Retrieved {len(gold_answers)} gold answers.")

    # 2. Model Setup
    model_id = "google/flan-t5-base"
    print(f"Loading closed-book LLM: '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()

    config_details = {
        "model_id": model_id,
        "architecture": "Seq2Seq / Flan-T5-Base",
        "parameters": "250M",
        "closed_book_prompt_template": "Answer the question: {question}",
        "decoding_parameters": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 50,
            "random_seed": 42
        },
        "retrieval_used": False,
        "corpus_access_allowed": False,
        "external_search_allowed": False
    }

    # Save configuration file
    with open(os.path.join(output_dir, "eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_details, f, indent=2)

    # 3. Closed-Book Generation & Evaluation
    print("Starting closed-book evaluation on 50 queries...")
    results = []
    correct_count = 0
    exact_match_count = 0

    t0 = time.time()
    for idx, q_obj in enumerate(pilot_queries):
        qid = q_obj["_id"]
        question = q_obj["text"]
        gold_ans = gold_answers.get(qid, "UNKNOWN")

        # Closed-book prompt ONLY (No context/retrieval)
        prompt = f"Answer the question: {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        model_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Assessment
        eval_metrics = compute_correctness(model_ans, gold_ans)
        if eval_metrics["is_correct"]:
            correct_count += 1
        if eval_metrics["exact_match"]:
            exact_match_count += 1

        record = {
            "query_index": idx + 1,
            "query_id": qid,
            "question": question,
            "model_answer": model_ans,
            "gold_answer": gold_ans,
            "correctness_assessment": {
                "is_correct": eval_metrics["is_correct"],
                "exact_match": eval_metrics["exact_match"],
                "f1_score": eval_metrics["f1_score"],
                "substring_match": eval_metrics["substring_match"]
            },
            "model_details": {
                "model_id": model_id,
                "decoding": "greedy (temp=0.0)"
            }
        }
        results.append(record)

    total_time = round(time.time() - t0, 2)
    correct_rate = round((correct_count / len(pilot_queries)) * 100, 2)
    exact_match_rate = round((exact_match_count / len(pilot_queries)) * 100, 2)

    # 4. Save JSON results
    json_output_path = os.path.join(output_dir, "closed_book_results.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_type": "Closed-Book QA (No Retrieval / No Context)",
            "total_queries": len(pilot_queries),
            "correct_count": correct_count,
            "correct_rate_percent": correct_rate,
            "exact_match_rate_percent": exact_match_rate,
            "evaluation_time_seconds": total_time,
            "model_config": config_details,
            "query_results": results
        }, f, indent=2, ensure_ascii=False)

    # 5. Generate Markdown Summary Report
    summary_md = f"""# 🤖 Closed-Book Evaluation Report: HotpotQA Pilot

**Evaluation Type:** Closed-Book QA (No Context / No Retrieval / No External Search)  
**Model:** `{model_id}` (Greedy Decoding, Temp=0.0, Seed=42)  
**Total Queries:** `{len(pilot_queries)}`  
**Overall Correct Rate:** **`{correct_rate}%`** (`{correct_count}/{len(pilot_queries)}` correct)  
**Exact Match (EM) Rate:** `{exact_match_rate}%` (`{exact_match_count}/{len(pilot_queries)}`)  
**Total Eval Time:** `{total_time}s`  

---

## ⚙️ Model & Configuration Details

```json
{json.dumps(config_details, indent=2)}
```

---

## 📋 Evaluation Results by Question

| # | Query ID | Question | Model Answer (Closed-Book) | Gold Answer | Assessment |
| :-: | :--- | :--- | :--- | :--- | :-: |
"""

    for r in results:
        status_icon = "✅ Correct" if r["correctness_assessment"]["is_correct"] else "❌ Incorrect"
        summary_md += f"| {r['query_index']} | `{r['query_id']}` | {r['question']} | `{r['model_answer']}` | `{r['gold_answer']}` | {status_icon} |\n"

    summary_md_path = os.path.join(output_dir, "closed_book_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nClosed-book evaluation completed in {total_time}s.")
    print(f"Overall Correct Rate: {correct_rate}% ({correct_count}/{len(pilot_queries)})")
    print(f"Exact Match Rate: {exact_match_rate}%")
    print(f"Results saved to '{json_output_path}' and '{summary_md_path}'")

if __name__ == "__main__":
    run_closed_book_eval()
