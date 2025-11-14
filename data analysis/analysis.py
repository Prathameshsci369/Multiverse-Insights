#!/usr/bin/env python3
"""
Analysis functions for the analysis pipeline with integrated model manager.
"""

import logging
import os
import gc
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PHI4_MAX_CONTEXT, FINAL_REPORT_TOKENS
from utils import estimate_tokens, _iso_now

# Model Manager Class
class QwenModelManager:
    """
    Model manager for Qwen2.5 7B model with streaming support.
    """
    def __init__(self, model_path, max_context=PHI4_MAX_CONTEXT):
        self.model_path = model_path
        self.max_context = max_context
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5 model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logging.info(f"Loading Qwen2.5 model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt, max_tokens=3000, temperature=0.0, stream=False):
        """Generate text from the model with optional streaming"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            if stream:
                return self._generate_stream(inputs, max_tokens, temperature)
            else:
                return self._generate_non_stream(inputs, max_tokens, temperature)
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return f"ERROR: Generation failed - {str(e)}"
    
    def _generate_non_stream(self, inputs, max_tokens, temperature):
        """Non-streaming generation"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response
    
    def _generate_stream(self, inputs, max_tokens, temperature):
        """Streaming generation"""
        with torch.no_grad():
            # Generate tokens one by one
            for token in self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                stream=True
            ):
                # Decode the new token
                if isinstance(token, torch.Tensor):
                    new_tokens = token[0][inputs["input_ids"].shape[1]:]
                else:
                    new_tokens = token
                
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                yield text
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache() if hasattr(torch, 'cuda') and torch.cuda.is_available() else None

# Process Batch Function
def process_batch(model_manager, batch, batch_id):
    """
    Processes a single batch using the Qwen2.5 model with streaming output.
    
    Args:
        model_manager: Instance of QwenModelManager
        batch: List of text strings to analyze
        batch_id: Identifier for the batch
        
    Returns:
        Dictionary with batch_id and result
    """
    # Join batch texts and clean
    joined_text = " ".join(text.strip() for text in batch if text)
    
    # Ultra-conservative token management
    prompt_overhead = 400  # System message and prompt structure
    expected_response = 4000  # Response generation
    safety_margin = 1000  # Additional safety buffer
    max_safe_tokens = PHI4_MAX_CONTEXT - prompt_overhead - safety_margin
    
    # Pre-validate token count
    total_tokens = estimate_tokens(joined_text)
    if total_tokens > max_safe_tokens:
        logging.warning(f"Batch {batch_id} requires {total_tokens} tokens, exceeding safe limit of {max_safe_tokens}")

    # Estimate total tokens
    total_tokens = estimate_tokens(joined_text)

    if total_tokens > max_safe_tokens:
        # Calculate a more conservative truncation
        safe_ratio = (max_safe_tokens - 1000) / total_tokens  # Extra 1000 token safety margin
        cutoff_chars = int(len(joined_text) * safe_ratio)
        
        # Try to cut at a sentence boundary
        if cutoff_chars < len(joined_text):
            # Look for last sentence boundary
            for marker in [". ", "! ", "? ", "\n", ". \n"]:
                last_boundary = joined_text[:cutoff_chars].rfind(marker)
                if last_boundary != -1 and last_boundary > cutoff_chars * 0.8:  # At least 80% of content
                    cutoff_chars = last_boundary + 1
                    break
                    
        joined_text = joined_text[:cutoff_chars]
        logging.warning(f"[process_batch] Batch {batch_id} ({total_tokens} tokens) truncated to fit context window.")

    prompt = f"""Analyze the following text and provide a structured summary, sentiment, and key topics.

Text:
{joined_text}

---
Report:
- Summary:
- Sentiment:
- Topics:
"""

    # Generate with streaming and collect the result
    result_chunks = []
    print(f"\n--- Processing Batch {batch_id} ---")
    
    try:
        for chunk in model_manager.generate(prompt, max_tokens=3000, stream=True):
            # Stream to terminal
            print(chunk, end="", flush=True)
            # Collect for result
            result_chunks.append(chunk)
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {e}")
        return {"batch_id": batch_id, "result": f"ERROR: {str(e)}"}
    
    # Join all chunks to get the complete result
    result = "".join(result_chunks)
    print("\n--- Batch Processing Complete ---\n")
    
    return {"batch_id": batch_id, "result": result}
