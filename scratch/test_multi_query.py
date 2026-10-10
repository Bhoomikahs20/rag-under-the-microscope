import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_multi_query():
    model_id = "google/flan-t5-base"
    print(f"Loading '{model_id}' for multi-query expansion...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    sample_questions = [
        "What character in the television special \"The Pound Puppies\" did the voice actor of Fred Jones from \"Scooby-Doo\" voice as?",
        "Edmundo \"Eddie\" Rodriguez was born in what Mexican state?",
        "What friends star's wedding was photographed by Robert Evans?"
    ]

    for q in sample_questions:
        prompt1 = f"Provide a keyword-focused search query for: {q}\nQuery:"
        prompt2 = f"Rephrase this question into a declarative search assertion: {q}\nAssertion:"
        
        in1 = tokenizer(prompt1, return_tensors="pt")
        in2 = tokenizer(prompt2, return_tensors="pt")
        
        out1 = model.generate(**in1, max_new_tokens=30, do_sample=False)
        out2 = model.generate(**in2, max_new_tokens=30, do_sample=False)
        
        var1 = tokenizer.decode(out1[0], skip_special_tokens=True).strip()
        var2 = tokenizer.decode(out2[0], skip_special_tokens=True).strip()
        
        print("="*75)
        print(f"Original Q : {q}")
        print(f"Variant 1  : {var1}")
        print(f"Variant 2  : {var2}")

if __name__ == "__main__":
    test_multi_query()
