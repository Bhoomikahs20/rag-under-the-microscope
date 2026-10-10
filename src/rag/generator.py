"""
FLAN-T5 RAG Generator Module
"""

import time
import torch
from typing import List, Dict, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils.logging_config import setup_logger

logger = setup_logger("generator")

class RAGGenerator:
    """FLAN-T5-Base Generator for Open-Book Context-Augmented QA."""
    
    def __init__(self, model_id: str = "google/flan-t5-base", seed: int = 42):
        self.model_id = model_id
        self.seed = seed
        torch.manual_seed(seed)
        logger.info(f"Loading generator model: '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Formats context passages and generates open-book RAG answer.
        
        Args:
            question: Question string
            retrieved_chunks: List of retrieved chunk dictionaries
            max_new_tokens: Max generated answer length
            
        Returns:
            Dict containing prompt, context, generated answer, and generation latency.
        """
        t0 = time.time()
        
        # Format context passages
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks):
            title = chunk.get("title", "")
            text = chunk.get("text", "")
            header = f"Passage {idx + 1} ({title})" if title else f"Passage {idx + 1}"
            context_blocks.append(f"{header}:\n{text}")
            
        context_str = "\n\n".join(context_blocks)
        
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer the question based ONLY on the context provided above:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
        generated_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        latency = round(time.time() - t0, 3)
        
        return {
            "prompt": prompt,
            "context": context_str,
            "generated_answer": generated_answer,
            "latency_seconds": latency
        }
