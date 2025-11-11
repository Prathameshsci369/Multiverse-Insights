#!/usr/bin/env python3
"""
RAG Chatbot for the text analysis application with enhanced long-term memory.
This module provides a conversational interface to query the analysis results
using the existing chunking and batching infrastructure, specifically configured for Qwen LLM.
"""

import os
import json
import pickle
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Updated LangChain imports to address deprecation warnings and compatibility issues
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate  # Import from langchain_core instead

# Local imports
from config import GGUF_PATH, N_GPU_LAYERS, N_CTX, CHAT_FORMAT
from utils import chunk_texts, estimate_tokens, get_timestamp

class RAGChatbot:
    def __init__(self, model_path: str = GGUF_PATH, vector_db_path: str = "temp_vector_db"):
        """
        Initialize the RAG chatbot with model and vector database paths.
        
        Args:
            model_path: Path to the GGUF model file
            vector_db_path: Path to store/load the vector database
        """
        self.model_path = model_path
        self.vector_db_path = vector_db_path
        self.memory_db_path = "conversation_memory.db"
        self.embeddings = None
        self.vector_db = None
        self.memory_db = None
        self.llm = None
        self.chunks = []
        self.chat_history = []  # Full conversation history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.question_count = 0  # Track the number of questions asked
        
    def initialize_embeddings(self):
        """Initialize the embedding model."""
        print(f"[{get_timestamp()}] Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print(f"[{get_timestamp()}] Embedding model initialized.")
    
    def initialize_llm(self):
        """Initialize the Qwen LLM for generation."""
        print(f"[{get_timestamp()}] Initializing Qwen LLM...")
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            # Qwen-specific configuration
            model_kwargs={"chat_format": CHAT_FORMAT},
            temperature=0.1,
            verbose=False,
            # Additional parameters for Qwen
            top_p=0.9,
            repeat_penalty=1.1,
        )
        print(f"[{get_timestamp()}] Qwen LLM initialized.")
    
    def process_chunks_from_json(self, json_path: str):
        """
        Process chunks from the JSON file using the existing chunking infrastructure.
        
        Args:
            json_path: Path to the JSON file containing the original data
        """
        print(f"[{get_timestamp()}] Processing chunks from {json_path}...")
        
        # Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text content
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Include title and selftext if they exist
                    if 'title' in item:
                        texts.append(f"TITLE: {item['title'].strip()}")
                    if 'selftext' in item:
                        texts.append(item['selftext'].strip())
        
        # Create chunks using the existing chunking function
        self.chunks = chunk_texts(texts, 1200, 100)  # Using the same chunk size and overlap as in config
        
        print(f"[{get_timestamp()}] Created {len(self.chunks)} chunks for vector database.")
    
    def create_vector_db(self):
        """Create a vector database from the processed chunks."""
        if not self.chunks:
            print(f"[{get_timestamp()}] No chunks available. Please process chunks first.")
            return
        
        print(f"[{get_timestamp()}] Creating vector database from {len(self.chunks)} chunks...")
        
        # Create text chunks and metadata
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(self.chunks):
            texts.append(chunk)
            metadatas.append({"chunk_id": i, "source": "original_data"})
        
        # Create the vector database
        self.vector_db = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        
        # Save the vector database
        self.vector_db.save_local(self.vector_db_path)
        print(f"[{get_timestamp()}] Vector database created and saved to {self.vector_db_path}.")
    
    def initialize_memory_db(self):
        """Initialize or load the conversation memory database."""
        print(f"[{get_timestamp()}] Initializing conversation memory database...")
        
        # Check if memory database exists
        if os.path.exists(self.memory_db_path):
            try:
                self.memory_db = FAISS.load_local(
                    self.memory_db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Fix for the pickle warning
                )
                print(f"[{get_timestamp()}] Loaded existing conversation memory database.")
            except Exception as e:
                print(f"[{get_timestamp()}] Error loading memory database: {e}. Creating new one.")
                self.memory_db = FAISS.from_texts(
                    ["Initial conversation memory"], 
                    self.embeddings, 
                    metadatas=[{"session_id": "init", "type": "system", "timestamp": datetime.now().isoformat()}]
                )
        else:
            # Create a new memory database
            self.memory_db = FAISS.from_texts(
                ["Initial conversation memory"], 
                self.embeddings, 
                metadatas=[{"session_id": "init", "type": "system", "timestamp": datetime.now().isoformat()}]
            )
            print(f"[{get_timestamp()}] Created new conversation memory database.")
    
    def save_memory_db(self):
        """Save the conversation memory database."""
        if self.memory_db:
            self.memory_db.save_local(self.memory_db_path)
            print(f"[{get_timestamp()}] Saved conversation memory database.")
    
    def add_to_memory(self, user_input: str, assistant_response: str):
        """Add a conversation exchange to the memory database."""
        if not self.memory_db:
            return
        
        # Create text representations of the exchange
        user_text = f"User asked: {user_input}"
        assistant_text = f"Assistant responded: {assistant_response}"
        
        # Add to memory database with metadata
        self.memory_db.add_texts(
            [user_text, assistant_text],
            metadatas=[
                {
                    "session_id": self.session_id,
                    "type": "user",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "session_id": self.session_id,
                    "type": "assistant",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )
        
        # Save the updated memory database
        self.save_memory_db()
    
    def retrieve_relevant_memory(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant conversation history based on the query."""
        if not self.memory_db:
            return []
        
        # Search for relevant memories
        docs = self.memory_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def initialize(self, json_path: str):
        """Initialize all components of the RAG chatbot."""
        self.initialize_embeddings()
        self.process_chunks_from_json(json_path)
        self.create_vector_db()
        self.initialize_llm()
        self.initialize_memory_db()
    
    def generate_response_with_question_count(self, question: str):
        """Generate a response that includes an accurate count of questions asked."""
        # Check if the user is asking about the number of questions
        if "how many question" in question.lower() and ("ask" in question.lower() or "asked" in question.lower()):
            # Generate a response that lists all questions
            questions_list = "\n".join([
                f"{i+1}. {exchange['user']}" 
                for i, exchange in enumerate(self.chat_history)
            ])
            
            response = f"You have asked {len(self.chat_history)} questions so far:\n\n{questions_list}"
            
            # Stream this response
            print("Chatbot: ", end="", flush=True)
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.01)  # Small delay for streaming effect
            print()
            
            # Add this exchange to the chat history
            self.chat_history.append({"user": question, "assistant": response})
            self.add_to_memory(question, response)
            
            return response
        
        # For other questions, use the normal flow
        return self.query_with_streaming(question)
    
    def query_with_streaming(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question and stream the response using Qwen's chat format.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if not self.vector_db or not self.llm:
            print(f"[{get_timestamp()}] Vector database or LLM not initialized. Please run initialize() first.")
            return {"error": "Vector database or LLM not initialized"}
        
        # Increment question count
        self.question_count += 1
        
        # Retrieve relevant documents from the main vector database
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        
        # Retrieve relevant conversation history
        relevant_memories = self.retrieve_relevant_memory(question, k=3)
        memory_context = "\n\n".join(relevant_memories) if relevant_memories else "No relevant conversation history found."
        
        # Combine the documents into a context
        document_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create messages in the format expected by Qwen
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to previous conversation history. Use the provided context and conversation history to answer the user's question concisely and accurately."},
        ]
        
        # Add recent chat history to messages (last 3 exchanges for immediate context)
        for exchange in self.chat_history[-3:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current question with context
        messages.append({"role": "user", "content": f"""Document Context: {document_context}
            
Conversation History: {memory_context}

Current Question: {question}

Please provide a concise answer based on the document context and conversation history above."""})
        
        print(f"[{get_timestamp()}] Generating response...")
        
        # Generate the response with streaming using Qwen's chat format
        full_response = ""
        print("Chatbot: ", end="", flush=True)
        
        # Use the streaming method with Qwen's chat format
        for chunk in self.llm.client.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1
        ):
            content = chunk["choices"][0]["delta"].get("content", "") if "choices" in chunk and len(chunk["choices"]) > 0 else ""
            if content:
                print(content, end="", flush=True)
                full_response += content
        
        print()  # New line after the response
        
        # Update chat history
        self.chat_history.append({"user": question, "assistant": full_response})
        
        # Add to memory database
        self.add_to_memory(question, full_response)
        
        return {
            "answer": full_response,
            "source_documents": docs,
            "relevant_memories": relevant_memories,
            "question_count": self.question_count
        }
    
    def interactive_chat(self):
        """Start an interactive chat session with streaming responses."""
        print("\n" + "="*50)
        print("RAG Chatbot for Text Analysis Results (Qwen)")
        print("Type 'exit' to end the conversation.")
        print("Type 'history' to see the conversation summary.")
        print("Type 'clear' to clear the current session history.")
        print("Type 'reset' to reset the question count.")
        print("="*50 + "\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            elif user_input.lower() == 'history':
                self.summarize_conversation()
                continue
            elif user_input.lower() == 'clear':
                self.chat_history = []
                print("Chatbot: Current session history cleared.")
                continue
            elif user_input.lower() == 'reset':
                self.question_count = 0
                print("Chatbot: Question count reset.")
                continue
            
            # Check if the user is asking about the number of questions
            if "how many question" in user_input.lower() and ("ask" in user_input.lower() or "asked" in user_input.lower()):
                self.generate_response_with_question_count(user_input)
            else:
                self.query_with_streaming(user_input)
            
            print()  # Add a blank line for readability
    
    def summarize_conversation(self):
        """Generate a summary of the conversation history."""
        if not self.chat_history:
            print("Chatbot: No conversation history to summarize.")
            return
        
        # Create a summary prompt
        summary_prompt = "Please summarize the following conversation between a user and an assistant:\n\n"
        
        # Add all exchanges to the prompt
        for i, exchange in enumerate(self.chat_history):
            summary_prompt += f"Q{i+1}: {exchange['user']}\n"
            summary_prompt += f"A{i+1}: {exchange['assistant']}\n\n"
        
        summary_prompt += "Provide a concise summary of the main topics discussed and questions asked."
        
        # Create messages for the summary
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summary_prompt}
        ]
        
        print("Chatbot: ", end="", flush=True)
        
        # Generate the summary
        for chunk in self.llm.client.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1
        ):
            content = chunk["choices"][0]["delta"].get("content", "") if "choices" in chunk and len(chunk["choices"]) > 0 else ""
            if content:
                print(content, end="", flush=True)
        
        print()  # New line after the response

def create_rag_chatbot(json_path: str) -> RAGChatbot:
    """
    Create and initialize a RAG chatbot.
    
    Args:
        json_path: Path to the JSON file containing the original data
        
    Returns:
        Initialized RAGChatbot instance
    """
    chatbot = RAGChatbot()
    chatbot.initialize(json_path)
    return chatbot

def main():
    """Main function to run the RAG chatbot."""
    print("Initializing RAG Chatbot with Qwen and Long-Term Memory...")
    json_path = "/home/anand/Documents/data/reddit_search_output1.json"  # Default path from config
    chatbot = create_rag_chatbot(json_path)
    
    # Start interactive chat
    chatbot.interactive_chat()

if __name__ == "__main__":
    main()