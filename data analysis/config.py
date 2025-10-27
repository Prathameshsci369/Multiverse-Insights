#!/usr/bin/env python3
"""
Configuration constants for the analysis pipeline.
"""

# File paths
JSON_PATH = "/home/anand/Documents/data/reddit_search_output1.json"
#GGUF_PATH = "/home/anand/Downloads/Qwen2.5-7B-Instruct.Q5_K_M.gguf"
GGUF_PATH = "/home/anand/Downloads/phi4.gguf"
#GGUF_PATH = "/home/anand/Downloads/phi4.gguf"
#GGUF_PATH = "/home/anand/Downloads/Phi-3-mini-4k-instruct-fp16.gguf"
CACHE_FILE = "partial_results.pkl"
LOG_FILE = "analysis_log.txt"

# Model settings
PHI4_MAX_CONTEXT = 17384
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


# Model settings
N_GPU_LAYERS = 0  # Setting this to 0 to force CPU-only execution
N_CTX = 25000  # Set a very large context size to handle extensive input text
CHAT_FORMAT = "qwen"  # The chat template required for Qwen models
#!/usr/bin/env python3
"""
Configuration constants for the analysis pipeline.
"""

# File paths
JSON_PATH = "/home/anand/Documents/data/reddit_search_output1.json"
GGUF_PATH = "/home/anand/Downloads/Qwen2.5-7B-Instruct.Q5_K_M.gguf"
CACHE_FILE = "partial_results.pkl"
LOG_FILE = "analysis_log.txt"

# Model settings
N_GPU_LAYERS = 0  # Setting this to 0 to force CPU-only execution
N_CTX = 25000  # Set a very large context size to handle extensive input text
CHAT_FORMAT = "qwen"  # The chat template required for Qwen models

# Text processing settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
BATCH_SIZE_TOKENS = 4000
MAX_TOKENS_PER_BATCH = N_CTX - 7000
FINAL_REPORT_TOKENS = 2048  # Increased output limit
TRANSLATION_MAX_LENGTH = 512
TRANSLATION_RETRIES = 3

# Safety limits
MIN_TOKENS_RESERVE = 7000
SAFETY_MARGIN = 1000

# Performance settings
MAX_RESPONSE_TIME = 300  # Maximum time in seconds for model response

#reddit credentials
client_id = "YOUR CLIENT ID"
client_secret = "YOUR_CLIENT_SECRET"
username = "YOUR_REDDIT_USERNAME" # Ensure this is exactly your Reddit username
password = "YOUR_REDDIT_ACCOUNT_PASSWORD" 

#youtube apis key
API_KEYS = [
    "YOUR_YOUTUBE_APY_KEY1",
    "YOUR_YOUTUBE_APY_KEY2",
    "YOUR_YOUTUBE_APY_KEY3",
    "YOUR_YOUTUBE_APY_KEY14" # Corrected a potential typo in one key - PLEASE VERIFY YOUR KEYS ARE CORRECT
]
