"""
demo_03_simple_agent.py - Simple Agent with Tool Calling

The key insight: an LLM alone just generates text. Give it tools and a loop,
and it becomes an agent -- a system that can plan, act, observe results,
and adapt. This is the ReAct pattern (Reason + Act), used in DSPy, LangChain, etc.

The loop:
    1. PLAN  -- LLM receives task + available tools, decides what to do
    2. ACT   -- we execute the tool calls it requested
    3. OBSERVE -- tool results are appended to conversation history
    4. REFLECT -- LLM sees results, decides next action or declares done

Why this works: each iteration, the LLM sees the full trail of its decisions
and their outcomes. It's not just answering a question -- it's conducting an
investigation, using tools to gather evidence, then synthesizing a conclusion.

This is a learning version with simple hardcoded tools.
For real API tools see demo_04; for LLM-generated tools see demo_05.

"""

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI

# Configuration
API_KEY="your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 2048
TIMEOUT = 300

# Disable Chain-of-Thought. We want direct JSON, not reasoning traces.
# Reduces tokens, speeds up response, cleaner output for parsing.
EXTRA_BODY = {
    "chat_template_kwargs": {
        "enable_thinking": False,
        "preserve_thinking": False
    }
}

client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# =============================================================================
# TOOLS -- deterministic functions the agent can call
# =============================================================================
# Tools are pure Python functions. They run locally, instantly, with 100% accuracy.
# The LLM never executes code -- it only decides which tool to call and with
# what arguments. We then execute the real function and feed results back.
# This separation (LLM for planning, Python for execution) is the whole point.

def reverse_complement(sequence: str) -> dict:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    try:
        rc = ''.join(complement[b] for b in reversed(sequence.upper()))
        return {"success": True, "result": rc}
    except KeyError as e:
        return {"success": False, "error": f"Invalid base: {e}"}


def count_bases(sequence: str) -> dict:
    """Count occurrences of each base and calculate GC content."""
    try:
        counts = {b: sequence.upper().count(b) for b in 'ATGC'}
        total = len(sequence)
        gc_content = (counts['G'] + counts['C']) / total * 100 if total > 0 else 0
        return {"success": True, "counts": counts, "total": total, "gc_content": round(gc_content, 2)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def explain_concept(concept: str) -> dict:
    """Get explanation of a bioinformatics concept."""
    explanations = {
        "gc_content": "GC content is the percentage of G and C bases in DNA.",
        "reverse_complement": "DNA sequence read backwards with complements (A-T, G-C).",
    }
    return {"success": True, "explanation": explanations.get(concept.lower(), f"No explanation for '{concept}'.")}


def find_motif(sequence: str, motif: str) -> dict:
    """Find all occurrences of a motif in a DNA sequence (including overlapping)."""
    try:
        positions = []
        start = 0
        seq_upper = sequence.upper()
        motif_upper = motif.upper()

        while True:
            pos = seq_upper.find(motif_upper, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1

        return {"success": True, "motif": motif, "count": len(positions), "positions": positions}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Tool Registry -- the agent's "menu" of available capabilities.
# Keys are tool names the LLM can reference. Values are the actual functions.
# This indirection lets us add/remove tools without changing agent logic.
TOOLS = {
    "reverse_complement": reverse_complement,
    "count_bases": count_bases,
    "explain_concept": explain_concept,
    "find_motif": find_motif,
}


# Agent Response Models
# =============================================================================
# AGENT RESPONSE MODELS -- what the LLM decides each iteration
# =============================================================================
# The agent doesn't just output text. It outputs a structured decision:
#   - tool_calls: which tools to invoke and why
#   - done: whether the task is complete
#   - final_answer: the answer (only if done=True)
#
# This is the decision interface. The LLM fills in these fields, we execute.
# Instructor validates the structure so we always get parseable output.

class ToolCall(BaseModel):
    """One tool invocation the agent wants to perform."""

    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(description="Arguments to pass to the tool")
    # reasoning forces the LLM to explain its choice. This improves accuracy
    # because the model must articulate why it's picking a tool before committing.
    reasoning: str = Field(description="Why this tool is being called")


class AgentResponse(BaseModel):
    """The agent's decision at each loop iteration."""

    tool_calls: list[ToolCall] = Field(description="List of tool calls to execute")
    final_answer: str | None = Field(description="Final answer if task is complete")
    # done is the critical control signal. The agent decides when it has enough.
    # This is what makes it agentic -- the LLM controls the loop, not us.
    done: bool = Field(description="Whether the task is complete")


# Agent Class
# =============================================================================
# AGENT CLASS -- the plan-act-observe loop
# =============================================================================
# The agent is just a class that wraps: system prompt + tool registry + loop.
# The system prompt describes available tools and sets behavioral constraints.
# The loop sends context to the LLM, executes its decisions, feeds back results.

class SimpleAgent:
    """LLM agent: receives a task, decides which tools to use, iterates until done."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

        # The system prompt is the agent's instruction set. It lists tools
        # and sets rules. The LLM reads this every iteration, so keeping it
        # concise matters -- it eats into the context window.
        self.system_prompt = """You are a bioinformatics assistant with access to tools.

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
        """Run the agent loop: plan -> act -> observe -> reflect -> repeat.

        Each iteration:
        1. Send full conversation history to the LLM
        2. LLM returns AgentResponse with tool calls or final answer
        3. Execute tool calls, collect observations
        4. Append tool calls + results to conversation history
        5. Repeat until done=True or max_iterations exhausted
        """
        # Conversation history starts with system prompt + user task.
        # Each iteration appends more messages, so the LLM sees the full trail.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        for iteration in range(self.max_iterations):
            print(f"  [Iteration {iteration + 1}]")

            # PLAN: ask the LLM what to do next.
            # It sees the full conversation -- the original task, previous tool
            # calls, and their results. It decides: call more tools or finish.
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=AgentResponse,
                messages=messages
            )

            # If the agent says it's done, return the final answer.
            if response.done and response.final_answer:
                print(f"  [Done in {iteration + 1} iterations]")
                return response.final_answer

            # ACT: execute each tool call the agent requested.
            observations = []
            for tool_call in response.tool_calls:
                print(f"    Tool: {tool_call.tool_name}({tool_call.arguments})")
                print(f"    Reason: {tool_call.reasoning}")

                if tool_call.tool_name not in TOOLS:
                    # Agent hallucinated a tool name -- tell it so it can recover.
                    obs = {"error": f"Unknown tool: {tool_call.tool_name}"}
                else:
                    # Execute the real function. Result is deterministic Python.
                    result = TOOLS[tool_call.tool_name](**tool_call.arguments)
                    obs = result

                observations.append(obs)
                print(f"    Result: {obs}")

            # OBSERVE: append the agent's actions and their results to history.
            # Next iteration, the LLM sees: "I called X, got result Y, now..."
            messages.append({"role": "assistant", "content": str(response.tool_calls)})
            messages.append({"role": "user", "content": f"Tool results: {observations}"})

        # Safety bound -- if the agent loops too many times, bail out.
        return f"Task incomplete after {self.max_iterations} iterations."

# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 03: Simple Agent with Tools")
    print("=" * 60)
    print()

    agent = SimpleAgent(max_iterations=5)

    # Test 1: Multi-step task -- the agent must call two different tools.
    # This is the key test: does the agent plan across iterations?
    # Iteration 1: calls count_bases -> gets GC% result
    # Iteration 2: calls reverse_complement -> gets RC result
    # Iteration 3: combines both results into final answer
    print("--- Test 1: GC Content + Reverse Complement ---")
    task1 = "What is the GC content of ATGCATGCATGC, and what is its reverse complement?"
    print(f"Task: {task1}\n")
    result1 = agent.run(task1)
    print(f"\nFinal Answer: {result1}\n")

    # Test 2: Single tool -- simple lookup, should finish in 1-2 iterations.
    print("--- Test 2: Concept Explanation ---")
    task2 = "Explain what GC content means"
    print(f"Task: {task2}\n")
    result2 = agent.run(task2)
    print(f"\nFinal Answer: {result2}\n")

    # Test 3: Tool with specific arguments -- tests the agent's ability to
    # extract the right parameters from natural language.
    print("--- Test 3: Motif Finding ---")
    task3 = "Find all ATG start codons in the sequence ATGCATGCATGC"
    print(f"Task: {task3}\n")
    result3 = agent.run(task3)
    print(f"\nFinal Answer: {result3}\n")

    print("=" * 60)
    print("Demo Summary:")
    print("  - Agent architecture: system prompt + tool loop")
    print("  - 4 bioinformatics tools available")
    print("  - Agent decides which tool to call")
    print("  - Multi-step reasoning demonstrated (Test 1 used 2 tools)")
    print()
    print("Next: demo_04 for real API tools (UniProt, PubMed, BLAST)")
    print("=" * 60)
