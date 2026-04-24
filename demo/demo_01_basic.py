"""
demo_01_basic.py - Basic LLM API calls with denbi-llm-api

Demonstrates fundamental LLM interactions: API setup, chat completions,
and bioinformatics prompts. Think of this as "Hello World" for LLM APIs.

"""

import requests

# API Configuration
# Note: In production, use environment variables (os.environ.get("API_KEY"))

API_KEY = "your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "gpt-oss-120b"


def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Make a basic LLM API call.

    Args:
        prompt: Question or instruction for the LLM
        temperature: Randomness control (0.0 = focused, 1.0 = creative)

    Returns:
        LLM's text response
    """
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 01: Basic LLM Calls")
    print("=" * 60)
    print(f"API: {API_BASE}")
    print(f"Model: {MODEL}\n")

    # Test 1: Concept explanation
    print("--- Test 1: Concept Explanation ---")
    result = call_llm("What is a reverse complement in DNA? Explain in 2 sentences.")
    print(f"Q: What is a reverse complement in DNA?")
    print(f"A: {result}\n")

    # Test 2: Calculation (note: LLMs can make arithmetic errors)
    print("--- Test 2: GC Content Calculation ---")
    result = call_llm("What is the GC content of ATGCATGC? Give only the number.")
    print(f"Q: What is the GC content of ATGCATGC?")
    print(f"A: {result}\n")

    # Test 3: Sequence transformation
    print("--- Test 3: Sequence Transformation ---")
    result = call_llm("Reverse complement this DNA sequence: ATGCATGC. Give only the sequence.")
    print(f"Q: Reverse complement of ATGCATGC?")
    print(f"A: {result}\n")

    # Test 4: Information extraction
    print("--- Test 4: Information Extraction ---")
    text = "The BRCA1 gene on chromosome 17 has 24 exons and is associated with breast cancer."
    result = call_llm(f"Extract gene name, chromosome, and exon count from: '{text}'")
    print(f"Text: {text}")
    print(f"Extracted: {result}\n")

    print("=" * 60)
    print("Key Observations:")
    print("  - LLMs can answer questions and transform text")
    print("  - Outputs are unstructured and may contain errors")
    print("  - Next: demo_02 shows structured outputs with validation")
    print("=" * 60)
