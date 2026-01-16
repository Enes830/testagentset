"""
Configuration file for RAG System
Contains default settings and API credentials
"""

# API Configuration
AGENTSET_NAMESPACE_ID = ""
AGENTSET_API_KEY = ""
OPENAI_API_KEY = ""

# RAG Settings
TOP_K = 10  # Number of documents to retrieve
MIN_SCORE = 0.5  # Minimum relevance score (0-1)

# System Prompt for RAG responses
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on the following context.
If you cannot find the answer in the context, say so clearly.

Context:
{context}"""
