import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import HybridSearch
from lib.utils import load_dataset, load_movies

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "nvidia/nemotron-3.5-super-vl:free"


def rag_command(query: str):
    movies = load_movies()
    searcher = HybridSearch(movies)

    docs = searcher.rrf_search(query, limit=5)

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    print("Submitting request to LLM...")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    llm_response = response.choices[0].message.content
    return {"search_results": docs, "llm_response": llm_response}


def summarize_command(query, limit):
    movies = load_movies()
    searcher = HybridSearch(movies)

    docs = searcher.rrf_search(query, limit=5)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""
    print("Submitting request to LLM...")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    llm_response = response.choices[0].message.content
    return {"search_results": docs, "llm_response": llm_response}


def citations_command(query, limit):
    movies = load_movies()
    searcher = HybridSearch(movies)

    docs = searcher.rrf_search(query, limit=5)
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    print("Submitting request to LLM...")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    llm_response = response.choices[0].message.content
    return {"search_results": docs, "llm_response": llm_response}


def question_command(question, limit):
    movies = load_movies()
    searcher = HybridSearch(movies)

    context = searcher.rrf_search(question, limit=5)
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    print("Submitting request to LLM...")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    llm_response = response.choices[0].message.content
    return {"search_results": context, "llm_response": llm_response}
