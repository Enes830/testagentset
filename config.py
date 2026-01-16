"""
Configuration file for RAG System
Contains default settings and API credentials
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# API Configuration
AGENTSET_NAMESPACE_ID = "ns_cmkgncf8s000104l5l1i4rfq7"
AGENTSET_API_KEY = os.getenv("AGENTSET_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG Settings
TOP_K = 10  # Number of documents to retrieve
MIN_SCORE = 0.6  # Minimum relevance score (0-1)

# System Prompt for RAG responses
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on the following context.
If you cannot find the answer in the context, say so clearly.

Context:
{context}"""
