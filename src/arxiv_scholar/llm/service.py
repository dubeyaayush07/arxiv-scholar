import os
import json
import logging
from typing import List, AsyncGenerator, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1", model: str = "anthropic/claude-3.5-haiku"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.model = model
        
        if self.api_key:
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.client = None
            logger.warning("No API Key provided for LLMService. API calls will fail or fall back.")

    async def generate_hyde_abstract(self, query: str) -> str:
        if not self.client:
            raise ValueError("LLM client not initialized.")
            
        prompt = f"""
        Please write a brief academic abstract that directly answers the following query. 
        Do not use conversational filler, just write the abstract.
        Query: {query}
        """
        logger.info(f"Requesting HyDE abstract from LLM ({self.model}) for query: '{query}'")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    async def decompose_query(self, query: str) -> List[str]:
        if not self.client:
            logger.warning("No LLM client configured for DECOMPOSE. Falling back.")
            return [query]
            
        prompt = f"""
        Decompose this query into independent, fully contextualized sub-queries. 
        Each sub-query must be able to stand completely on its own for a search engine.
        For example: "Accuracy of BERT vs GPT-3" -> ["What is the accuracy of BERT?", "What is the accuracy of GPT-3?"]
        
        Query: {query}
        
        Return ONLY valid JSON in this exact format:
        {{
            "sub_queries": ["sub query 1", "sub query 2"]
        }}
        """
        logger.info(f"Requesting query decomposition from LLM ({self.model}) for query: '{query}'")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Safely strip markdown formatting if the model included it despite the json_object constraint
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return data.get("sub_queries", [query])
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON for decomposition: {e} | Content: {content}")
            return [query]

    async def stream_synthesis(self, query: str, context: str) -> AsyncGenerator[Any, None]:
        if not self.client:
            raise ValueError("LLM client not initialized.")
            
        prompt = f"""You are an academic research assistant. 
Answer the user's query comprehensively based ONLY on the provided context chunks. 
If the answer is not in the context, state that clearly. Cite your sources where applicable using the provided Source URL (e.g. [https://arxiv.org/abs/2010.05432]).

CRITICAL: Return ONLY the final answer. Do NOT use conversational filler (e.g. "Based on the provided context..."). Start your answer immediately.

Context:
{context}

Query: {query}
"""
        logger.info(f"Starting LLM synthesis stream ({self.model}) for query: '{query}'")
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True
        )
        
        async for chunk in stream:
            yield chunk
