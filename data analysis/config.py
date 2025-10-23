#!/usr/bin/env python3
"""
Configuration constants for the analysis pipeline.
"""

# File paths
JSON_PATH = "/home/anand/Documents/data/reddit_search_output1.json"
#GGUF_PATH = "/home/anand/Downloads/Qwen2.5-7B-Instruct.Q5_K_M.gguf"
GGUF_PATH = "/home/anand/Downloads/phi4.gguf"
#GGUF_PATH = "/home/anand/Downloads/mistral-7b-v0.1.Q4_K_S.gguf"
CACHE_FILE = "partial_results.pkl"
LOG_FILE = "analysis_log.txt"

# Model settings
PHI4_MAX_CONTEXT = 20384
NUM_MODELS = 1
MAX_WORKERS = 8
MODEL_THREADS = 6

# Text processing settings
CHUNK_SIZE = 1200  # Reduced chunk size for better management
CHUNK_OVERLAP = 100  # Reduced overlap
BATCH_SIZE_TOKENS = 4000  # Even more conservative batch size
MAX_TOKENS_PER_BATCH = PHI4_MAX_CONTEXT - 7000  # Reserve 7000 tokens for overhead
FINAL_REPORT_TOKENS = 6000  # Reduced final report tokens
TRANSLATION_MAX_LENGTH = 512
TRANSLATION_RETRIES = 3

# Safety limits
MIN_TOKENS_RESERVE = 7000  # Minimum tokens to reserve for system/prompt/response
SAFETY_MARGIN = 1000  # Additional safety margin for token calculations
