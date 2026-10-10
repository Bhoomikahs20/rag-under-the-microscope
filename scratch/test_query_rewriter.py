import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_rewriter():
    model_id = "google/flan-t5-base"
    print(f"Loading '{model_id}' for query rewriting...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    sample_questions = [
        "What character in the television special \"The Pound Puppies\" did the voice actor of Fred Jones from \"Scooby-Doo\" voice as?",
        "Edmundo \"Eddie\" Rodriguez was born in what Mexican state?",
        "What friends star's wedding was photographed by Robert Evans?",
        "What is the prequel to the show where Rose Eleanor Arbuthnot-Leslie stars as Maia Rindell?"
    ]

    for q in sample_questions:
        prompt = (
            "Rewrite and expand the following question into a clear search query for document retrieval by highlighting key entities, titles, and relationships:\n"
            f"Original Question: {q}\n"
            "Search Query:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("="*75)
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}")

if __name__ == "__main__":
    test_rewriter()
