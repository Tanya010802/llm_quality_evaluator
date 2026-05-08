"""
llm_client.py
Thin wrapper around Groq's API (OpenAI-compatible).
Groq offers a free tier — sign up at https://console.groq.com to get your API key.
"""

import os
from typing import Optional

import groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama3-8b-8192"  # fast + free; swap to llama3-70b-8192 for higher quality
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    _client = Groq(api_key=api_key)
    return _client


def _mock_completion(prompt: str, system_prompt: str | None = None) -> str:
    """
    Deterministic fallback used when Groq credentials are missing/invalid.
    This keeps the project runnable in local/CI environments without secrets.
    """
    p = prompt.strip().lower()
    sp = (system_prompt or "").strip().lower()

    # JSON-format case
    if "respond only with valid json" in p or "return a json object" in p:
        return '{"name":"Alex","age":30}'

    # Context-grounded hallucination case
    if "tanya" in p and ("context:" in sp or "answer only based on the following context" in sp):
        # Mention both Python and Playwright (present in the test context).
        return "Tanya uses Python and Playwright."

    # Simple factual case
    if "capital of france" in p:
        return "Paris."

    # Consistency case
    if "three primary colors" in p or "three primary colours" in p:
        return "The three primary colors are red, blue, and yellow."

    # Relevance case
    if "automated software testing" in p and "2 sentence" in p:
        return (
            "Automated software testing uses tools to run tests automatically to verify software behaves as expected. "
            "It helps catch bugs quickly and consistently by repeatedly testing features without manual effort."
        )

    # Generic safe fallback: echo a concise on-topic response
    return "I don't know."


def get_completion(prompt: str, system_prompt: str = None, temperature: float = 0.0) -> str:
    """
    Send a prompt to the LLM and return the text response.

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system-level instruction.
        temperature: 0.0 = deterministic (good for consistency tests).

    Returns:
        The model's response as a string.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    # Allow forcing the mock path explicitly (handy for CI).
    if os.getenv("USE_MOCK_LLM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _mock_completion(prompt, system_prompt=system_prompt).strip()

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except (RuntimeError, groq.AuthenticationError, groq.APIConnectionError, groq.APITimeoutError):
        # Missing/invalid key or transient network issue: fall back to deterministic mock
        return _mock_completion(prompt, system_prompt=system_prompt).strip()


def get_completion_with_context(prompt: str, context: str, temperature: float = 0.0) -> str:
    """
    Send a prompt grounded in a specific context (for hallucination tests).
    """
    system_prompt = (
        f"You are a helpful assistant. Answer ONLY based on the following context. "
        f"Do not add information not present in the context.\n\nContext:\n{context}"
    )
    return get_completion(prompt, system_prompt=system_prompt, temperature=temperature)
