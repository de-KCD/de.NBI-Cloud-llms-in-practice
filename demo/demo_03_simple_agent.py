"""
demo_03_agent.py - Test simple agent with tools

This is a SIMPLIFIED version of the agent demo - perfect for learning!

What this demo shows:
- Basic agent architecture (system prompt + tool loop)
- How tools are defined and registered
- Agent decision-making (call tools vs give final answer)
- Conversation history management

DIFFERENCE FROM OTHER DEMOS:
- demo_04: Real API tools (BLAST, UniProt, PubMed)
- demo_05: LLM-generated tools (no hardcoded logic)
- demo_03: Simple hardcoded tools (learning version)

START HERE if you're new to agents!
"""

# =============================================================================
# IMPORTS
# =============================================================================

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================
# API credentials - hardcoded for demo (use env vars in production!)

API_KEY = "your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 8192

# Timeout for API calls (seconds)
TIMEOUT = 300

# =============================================================================
# LLM CLIENT SETUP
# =============================================================================
# instructor forces the LLM to return structured JSON matching our Pydantic models
# This is CRITICAL for reliable agent behavior!

client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# Tools are Python functions the agent can call to get REAL information
#
# Why tools matter:
# - LLMs can't actually COMPUTE (they're language models, not calculators)
# - Tools let the LLM interact with the real world
# - Each tool is like a "superpower" you give the agent
#
# Tool design principles:
# 1. Clear, descriptive names
# 2. Typed parameters
# 3. Return dict with 'success' status
# 4. Handle errors gracefully
# =============================================================================

def reverse_complement(sequence: str) -> dict:
    """
    Return the reverse complement of a DNA sequence.

    DNA has two strands that pair: A↔T, G↔C
    Reverse complement = read backwards + swap bases

    Example: ATGC → GCAT
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    try:
        rc = ''.join(complement[b] for b in reversed(sequence.upper()))
        return {"success": True, "result": rc}
    except KeyError as e:
        return {"success": False, "error": f"Invalid base: {e}"}


def count_bases(sequence: str) -> dict:
    """
    Count occurrences of each base and calculate GC content.

    GC content = (G + C) / total × 100
    High GC% = more stable DNA (3 H-bonds vs 2 for AT)
    """
    try:
        counts = {b: sequence.upper().count(b) for b in 'ATGC'}
        total = len(sequence)
        gc_content = (counts['G'] + counts['C']) / total * 100 if total > 0 else 0
        return {"success": True, "counts": counts, "total": total, "gc_content": round(gc_content, 2)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def explain_concept(concept: str) -> dict:
    """
    Get explanation of a bioinformatics concept.

    Simple lookup table - in production, might call LLM or query database.
    """
    explanations = {
        "gc_content": "GC content is the percentage of G and C bases in DNA.",
        "reverse_complement": "DNA sequence read backwards with complements (A-T, G-C).",
    }
    return {"success": True, "explanation": explanations.get(concept.lower(), f"No explanation for '{concept}'.")}


def find_motif(sequence: str, motif: str) -> dict:
    """
    Find all occurrences of a motif in a DNA sequence.

    A motif is a short DNA pattern with biological significance:
    - ATG = Start codon (where proteins begin)
    - TATAAA = TATA box (promoter element)
    - AATAAA = Poly-A signal (transcription termination)

    This finds ALL positions including overlapping matches.
    Example: "AAA" contains "AA" at positions 0 AND 1.

    Args:
        sequence: DNA sequence to search
        motif: Pattern to find (e.g., "ATG" for start codons)

    Returns:
        dict with success, motif, count, and positions (0-based)
    """
    try:
        positions = []
        start = 0
        seq_upper = sequence.upper()
        motif_upper = motif.upper()

        # Find all occurrences (including overlapping)
        while True:
            pos = seq_upper.find(motif_upper, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1  # Move by 1 to catch overlaps

        return {
            "success": True,
            "motif": motif,
            "count": len(positions),
            "positions": positions
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOLS REGISTRY
# =============================================================================
# Maps tool names to actual functions
# Agent says "call count_bases" → we look it up here and execute
#
# START HERE when adding new tools:
# 1. Define the function above
# 2. Add it to this dictionary
# 3. Update the system prompt below

TOOLS = {
    "reverse_complement": reverse_complement,
    "count_bases": count_bases,
    "explain_concept": explain_concept,
    "find_motif": find_motif,  # NEW: Find DNA motifs
}


# =============================================================================
# AGENT RESPONSE MODELS
# =============================================================================
# Pydantic models define the STRUCTURE of agent responses
# instructor ensures LLM returns valid JSON matching these models
# =============================================================================

class ToolCall(BaseModel):
    """Represents a tool call from the agent."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(description="Arguments to pass to the tool")
    reasoning: str = Field(description="Why this tool is being called")


class AgentResponse(BaseModel):
    """Agent's response at each iteration."""
    tool_calls: list[ToolCall] = Field(description="List of tool calls to execute")
    final_answer: str | None = Field(description="Final answer if task is complete")
    done: bool = Field(description="Whether the task is complete")


# =============================================================================
# AGENT CLASS
# =============================================================================
# The agent runs a loop:
# 1. Send task + history to LLM
# 2. LLM decides: call tools OR give final answer
# 3. If tools: execute them, add results to history, repeat
# 4. If done: return final answer
# =============================================================================

class SimpleAgent:
    """A simple LLM agent with tool-calling capability."""

    def __init__(self, max_iterations: int = 5):
        """
        Initialize the agent.

        Args:
            max_iterations: Maximum tool-calling rounds before giving up
        """
        self.max_iterations = max_iterations

        # System prompt = agent's "brain" - defines role, tools, and rules
        self.system_prompt = """You are a bioinformatics assistant. You have access to tools.

Available tools:
- reverse_complement(sequence: str) - Get reverse complement of DNA
- count_bases(sequence: str) - Count bases and calculate GC%
- explain_concept(concept: str) - Explain bioinformatics concepts
- find_motif(sequence: str, motif: str) - Find DNA motifs (e.g., ATG, TATA box)

Rules:
1. Call tools when you need information
2. Set done=true when you have enough information
3. Never make up data - use tools
4. Use the RIGHT tool for each task
"""

    def run(self, task: str) -> str:
        """
        Execute the agent loop to complete a task.

        Args:
            task: The user's question/request

        Returns:
            str: Final answer from the agent
        """
        # Initialize conversation with system prompt + user task
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        # Main agent loop
        for iteration in range(self.max_iterations):
            print(f"  [Iteration {iteration + 1}]")

            # Get LLM's decision
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                response_model=AgentResponse,
                messages=messages
            )

            # Check if done
            if response.done and response.final_answer:
                print(f"  [Done in {iteration + 1} iterations]")
                return response.final_answer

            # Execute tool calls
            observations = []
            for tool_call in response.tool_calls:
                print(f"    Tool: {tool_call.tool_name}({tool_call.arguments})")
                print(f"    Reason: {tool_call.reasoning}")

                if tool_call.tool_name not in TOOLS:
                    obs = {"error": f"Unknown tool: {tool_call.tool_name}"}
                else:
                    result = TOOLS[tool_call.tool_name](**tool_call.arguments)
                    obs = result

                observations.append(obs)
                print(f"    Result: {obs}")

            # Add to conversation history
            messages.append({"role": "assistant", "content": str(response.tool_calls)})
            messages.append({"role": "user", "content": f"Tool results: {observations}"})

        return f"Task incomplete after {self.max_iterations} iterations."


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 03: Simple Agent with Tools")
    print("=" * 60)
    print()

    agent = SimpleAgent(max_iterations=5)

    # Test 1: Basic analysis (requires 2 tools)
    print("--- Test 1: GC Content + Reverse Complement ---")
    task1 = "What is the GC content of ATGCATGCATGC, and what is its reverse complement?"
    print(f"Task: {task1}")
    print()
    result1 = agent.run(task1)
    print(f"\nFinal Answer: {result1}")
    print()

    # Test 2: Concept explanation
    print("--- Test 2: Concept Explanation ---")
    task2 = "Explain what GC content means"
    print(f"Task: {task2}")
    print()
    result2 = agent.run(task2)
    print(f"\nFinal Answer: {result2}")
    print()

    # Test 3: Motif finding (uses the new find_motif tool)
    print("--- Test 3: Motif Finding ---")
    task3 = "Find all ATG start codons in the sequence ATGCATGCATGC"
    print(f"Task: {task3}")
    print()
    result3 = agent.run(task3)
    print(f"\nFinal Answer: {result3}")
    print()

    # Summary
    print("=" * 60)
    print("DEMO 03 COMPLETE")
    print("=" * 60)
    print()
    print("What you saw:")
    print("  ✓ Agent architecture (system prompt + tool loop)")
    print("  ✓ 4 bioinformatics tools (reverse_complement, count_bases,")
    print("    explain_concept, find_motif)")
    print("  ✓ Agent decision-making (which tool to call)")
    print("  ✓ Multi-step reasoning (Test 1 used 2 tools)")
    print()
    print("Try it yourself:")
    print("  → Add your own tool (see tutorial for step-by-step)")
    print()
    print("Next steps:")
    print("  → demo_04: Real API tools (UniProt, PubMed, BLAST)")
    print("  → demo_05: LLM-generated tools (no hardcoded logic)")
    print("=" * 60)
