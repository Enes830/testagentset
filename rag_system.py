"""
RAG System - Retrieval Augmented Generation using Agentset and OpenAI
Simple class to handle document retrieval and AI-powered responses
"""

import requests
import logging
from openai import OpenAI as OpenAIClient

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    A simple RAG (Retrieval Augmented Generation) system.
    Retrieves relevant documents and generates responses using OpenAI.
    """

    def __init__(self, agentset_namespace_id: str, agentset_api_token: str, openai_api_key: str, system_prompt: str = None):
        """
        Initialize the RAG system with API credentials.

        Args:
            agentset_namespace_id: Agentset namespace ID
            agentset_api_token: Agentset API token
            openai_api_key: OpenAI API key
            system_prompt: Custom system prompt (optional)
        """
        logger.info("Initializing RAG System")
        
        self.agentset_namespace_id = agentset_namespace_id
        self.agentset_api_token = agentset_api_token
        self.openai_api_key = openai_api_key
        self.system_prompt = system_prompt

        # Initialize OpenAI client
        self.openai_client = OpenAIClient(api_key=openai_api_key)
        logger.debug("OpenAI client initialized")

        # Agentset API configuration
        self.agentset_base_url = "https://api.agentset.ai/v1"
        self.agentset_headers = {
            "Authorization": f"Bearer {agentset_api_token}",
            "Content-Type": "application/json",
        }
        logger.debug("Agentset API configured")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.5,
        rerank: bool = True,
        rerank_model: str = "zeroentropy:zerank-2",
    ) -> str:
        """
        Retrieve relevant documents from Agentset based on query.

        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum relevance score (0-1)
            rerank: Whether to rerank results
            rerank_model: Model to use for reranking

        Returns:
            Extracted context from retrieved documents
        """
        logger.info(f"Retrieving documents for query: '{query}' (top_k={top_k}, min_score={min_score})")
        
        search_url = f"{self.agentset_base_url}/namespace/{self.agentset_namespace_id}/search"

        payload = {
            "query": query,
            "topK": top_k,
            "rerank": rerank,
            "rerankLimit": top_k,
            "rerankModel": rerank_model,
            "filter": {},
            "minScore": min_score,
            "includeRelationships": False,
            "includeMetadata": True,
            "keywordFilter": "",
            "mode": "semantic",
        }

        response = requests.post(search_url, headers=self.agentset_headers, json=payload)

        if response.status_code != 200:
            logger.error(f"Agentset API error: {response.status_code} - {response.text}")
            raise Exception(f"Agentset API error: {response.status_code} - {response.text}")

        results = response.json()
        logger.debug(f"Agentset API returned {len(results.get('data', []))} results")

        # Extract context from search results
        context = ""
        if "data" in results:
            for item in results["data"]:
                if "text" in item:
                    context += item["text"] + "\n"

        logger.info(f"Extracted context of {len(context)} characters from {len(results.get('data', []))} documents")
        return context.strip()

    def generate_response(self, query: str, context: str, system_prompt: str = None) -> str:
        """
        Generate a response using OpenAI based on retrieved context.

        Args:
            query: User's question
            context: Retrieved context from documents
            system_prompt: Custom system prompt (optional, uses instance prompt if not provided)

        Returns:
            Generated response from OpenAI
        """
        logger.info(f"Generating response for query: '{query}'")
        
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if system_prompt is None:
            system_prompt = f"Answer questions based on the following context:\n\n{context}"
        else:
            # Replace {context} placeholder in system prompt
            system_prompt = system_prompt.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        result = response.choices[0].message.content
        logger.debug(f"Generated response of {len(result)} characters")
        return result

    def query(self, query: str, top_k: int = 10, min_score: float = 0.5) -> dict:
        """
        Execute a complete RAG pipeline: retrieve and generate.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            min_score: Minimum relevance score

        Returns:
            Dictionary with 'context' and 'response' keys
        """
        logger.info(f"Starting RAG query pipeline for: '{query}'")
        
        context = self.retrieve(query, top_k=top_k, min_score=min_score)
        response = self.generate_response(query, context)

        logger.info(f"RAG query completed successfully")
        
        return {
            "query": query,
            "context": context,
            "response": response,
        }
