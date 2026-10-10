import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open('data/processed/hotpotqa_pilot_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
model.eval()

doc = corpus[0]
prompt = f"Here is a document titled '{doc['title']}':\n{doc['text'][:300]}\n\nSummarize what this document is about in one sentence:"
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print('Title:', doc['title'])
print('Generated Context Summary:', summary)
