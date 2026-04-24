"""
demo_01_basic.py - Test basic LLM calls with denbi-llm-api

This is your first step into working with LLM APIs! This file shows you how to:
- Set up API credentials
- Make basic chat completion requests
- Ask the LLM to perform useful bioinformatics tasks

Think of this as "Hello World" for LLM-powered bioinformatics!
"""

# =============================================================================
# IMPORTS
# =============================================================================

import requests


# =============================================================================
# API CONFIGURATION
# =============================================================================
# These are the credentials needed to talk to the LLM API.
#
# SECURITY NOTE: In production, NEVER hardcode API keys like this!
# Instead, use environment variables: os.environ.get("API_KEY")
# We're hardcoding here just for demo/testing purposes.

# Your API key
API_KEY = "your-api-key"

# The base URL for the denbi-llm-api service
# Are you using de.NBI Cloud VM to do this tutorial?
# Uncomment and run ONLY ONE of them:
# YES:
# API_BASE="https://denbi-llm-api-internal.bihealth.org/v1"
# NO:
# API_BASE="https://denbi-llm-api.bihealth.org/v1"

# Which model to use for generating responses
# gpt-oss-120b is a fast and good general model
MODEL = "gpt-oss-120b"


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Make a basic LLM API call to generate a response.

    This is the heart of our LLM interaction - it sends your prompt to the
    API and returns the generated text response.

    Args:
        prompt: The question or instruction you want to ask the LLM
        temperature: Controls randomness (0.0 = focused, 1.0 = creative).
                    Lower is better for factual tasks.

    Returns:
        str: The LLM's text response

    Example:
        >>> response = call_llm("What is DNA?")
        >>> print(response)
        "DNA is a molecule that carries genetic instructions..."
    """

    # Send the API request
    # We're using requests.post() to send our data to the server
    response = requests.post(
        # The full endpoint URL for chat completions
        f"{API_BASE}/chat/completions",

        # Authentication header - proves we have permission to use the API
        headers={"Authorization": f"Bearer {API_KEY}"},

        # The request body - what we're asking the LLM to do
        json={
            "model": MODEL,                    # Which AI model to use
            "messages": [                      # Conversation history
                {"role": "user", "content": prompt}  # Our question
            ],
            "temperature": temperature,        # Creativity level
            "max_tokens": 1024,                # Max length of response
        },

        # Timeout after 300 seconds (5 minutes) to avoid hanging forever
        timeout=300
    )

    # Raise an exception if the API returned an error (e.g., 401, 500)
    # This helps us catch authentication issues, rate limits, etc.
    response.raise_for_status()

    # Parse the response and extract the generated text
    # The API returns a JSON object with a specific structure
    # We navigate through: response -> choices[0] -> message -> content
    return response.json()["choices"][0]["message"]["content"]


# =============================================================================
# MAIN EXECUTION
# =============================================================================
# This block runs when you execute the file directly:
#   python demo_01_basic.py
#
# It demonstrates the LLM with several test cases so you can see
# how different prompts and settings affect the output.

if __name__ == "__main__":
    # Print a nice header to make the output readable
    print("=" * 60)
    print("DEMO 01: Basic LLM Calls")
    print("=" * 60)
    print(f"API: {API_BASE}")
    print(f"Model: {MODEL}")
    print()

    # -------------------------------------------------------------------------
    # Test 1: Answer a Biology Question
    # -------------------------------------------------------------------------
    # Start with a simple concept question to see the LLM explain something.
    print("--- Test 1: Concept Explanation ---")
    result = call_llm("What is a reverse complement in DNA? Explain in 2 sentences.")
    print("Q: What is a reverse complement in DNA?")
    print(f"A: {result}")
    print()

    # -------------------------------------------------------------------------
    # Test 2: Perform a Calculation
    # -------------------------------------------------------------------------
    # Ask the LLM to calculate something concrete.
    # Note: LLMs are not calculators - they can make mistakes!
    # This is why demo_02 shows structured outputs for reliability.
    print("--- Test 2: GC Content Calculation ---")
    result = call_llm("What is the GC content (as a percentage) of the DNA sequence ATGCATGC? Give only the number.")
    print("Q: What is the GC content of ATGCATGC?")
    print(f"A: {result}")
    print()

    # -------------------------------------------------------------------------
    # Test 3: Transform Data
    # -------------------------------------------------------------------------
    # Ask the LLM to transform a sequence (reverse complement).
    # Again, this can be error-prone - demo_02 will show better approaches.
    print("--- Test 3: Sequence Transformation ---")
    result = call_llm("Reverse complement this DNA sequence: ATGCATGC. Give only the sequence, no explanation.")
    print("Q: Reverse complement of ATGCATGC?")
    print(f"A: {result}")
    print()

    # -------------------------------------------------------------------------
    # Test 4: Extract Information
    # -------------------------------------------------------------------------
    # Show the LLM as a parser - extracting structured info from text.
    # This sets up demo_02 (structured outputs with validation).
    print("--- Test 4: Information Extraction ---")
    text = "The BRCA1 gene on chromosome 17 has 24 exons and is associated with breast cancer."
    result = call_llm(f"Extract the gene name, chromosome, and exon count from this text: '{text}'. Return as: gene=NAME, chr=NUMBER, exons=NUMBER")
    print(f"Text: {text}")
    print(f"Extracted: {result}")
    print()

    # -------------------------------------------------------------------------
    # Summary: What This Shows
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("✓ LLMs can answer biology questions")
    print("✓ LLMs can perform simple calculations (but may be wrong)")
    print("✓ LLMs can transform data (but may make mistakes)")
    print("✓ LLMs can extract information (but format varies)")
    print()
    print("⚠ Problem: Outputs are unstructured and unreliable!")
    print("→ Next: demo_02 shows structured outputs with validation")
    print("=" * 60)
