"""
demo_05_llm_limitations.py - Understanding LLM Limitations

CRITICAL WARNING: Educational demo only. Illustrates what LLMs appear to do
vs. what they actually do.

Key insight: LLMs are pattern-matchers, not calculators. They generate text
that looks like calculations but do not perform actual computation.

This demo shows:
- How LLM-generated tools work (descriptions, not hardcoded functions)
- Why explanations are post-hoc rationalizations
- Why confidence scores are unreliable
- When this approach is dangerous vs. acceptable

For production: Always use hardcoded tools (demo_04) for real computation.

"""

import json
import time
import instructor
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from openai import OpenAI

# Configuration
API_KEY="your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 2048
TIMEOUT = 300.0

# Disable Chain-of-Thought. We want direct JSON, not reasoning traces.
# Reduces tokens, speeds up response, cleaner output for parsing.
# Why: with structured outputs, CoT thinking leaks into JSON fields as noise.
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
# Why Instructor: enforces schema on LLM output. Without it, the LLM returns
#   free text and you'd need regex/parsing to extract structured data.


# Tool Result Model
# Why include confidence despite it being unreliable: this demo teaches a lesson.
#   LLMs give false confidence -- 0.95 doesn't mean correct. Observing this
#   gap between stated and actual confidence is the whole point of this demo.
class ToolResult(BaseModel):
    """Result from an LLM-generated tool."""

    success: bool = Field(description="Whether the tool succeeded")
    result: Any = Field(description="The computed result - VERIFY for science!")
    explanation: str = Field(
        description="Plausible reasoning (NOT actual computation steps - LLMs don't calculate!)"
    )
    confidence: float = Field(
        description="LLM self-assessment (0-1) - NOT reliable",
        ge=0,
        le=1
    )


# Generated Tool Model
class GeneratedTool(BaseModel):
    """Metadata about an LLM-generated tool."""

    tool_name: str = Field(description="Name of the generated tool")
    description: str = Field(description="What this tool does")
    input_description: str = Field(description="Expected input format")
    output_description: str = Field(description="Expected output format")


def generate_and_execute_tool(task_description: str, input_data: Any) -> ToolResult:
    """Generate a tool on-the-fly and execute it via LLM pattern-matching."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            response_model=ToolResult,
            messages=[
                {
                    "role": "system",
                    "content": """You are a computational assistant.

HONESTY REQUIREMENT:
- LLMs don't actually calculate - you generate text that looks like calculations
- Be honest about uncertainty
- NEVER make up formulas or methods you're unsure about

GUIDELINES:
- For simple, common tasks, you've seen many examples
- For complex/long inputs, warn that verification is needed
- If unsure, recommend using hardcoded tools
"""
                },
                {
                    "role": "user",
                    "content": f"Task: {task_description}\nInput: {json.dumps(input_data, default=str)}\n\nPerform this computation and return the result."
                }
            ]
        )
        return response
    except Exception as e:
        return ToolResult(
            success=False,
            result=None,
            explanation=f"Tool generation failed: {str(e)}",
            confidence=0.0
        )


# LLM-Generated Tool Registry (descriptions only, no code)
# Why no code: the LLM is told what each tool does, then generates the logic.
#   This is flexible (any task the LLM understands) but unreliable.
#   Compare with demo_04's hardcoded tools where Python does the real work.
LLM_TOOLS = {
    "reverse_complement": {
        "description": "Get the reverse complement of a DNA sequence",
        "input": {"sequence": "DNA sequence string (e.g., 'ATGCATGC')"},
        "output": "Reverse complement sequence"
    },
    "count_bases": {
        "description": "Count each base (A,T,G,C) in a DNA sequence and calculate GC content",
        "input": {"sequence": "DNA sequence string"},
        "output": "Dict with counts and gc_content percentage"
    },
    "translate_dna": {
        "description": "Translate DNA sequence to protein using standard genetic code",
        "input": {"sequence": "DNA sequence", "frame": "Reading frame (1, 2, or 3)"},
        "output": "Protein sequence (single letter amino acid codes)"
    },
    "find_orfs": {
        "description": "Find open reading frames in DNA sequence",
        "input": {"sequence": "DNA sequence", "min_length": "Minimum ORF length in nucleotides"},
        "output": "List of ORFs with start, end, frame, length"
    },
    "calculate_melting_temp": {
        "description": "Calculate melting temperature (Tm) of DNA sequence using Wallace rule",
        "input": {"sequence": "DNA sequence"},
        "output": "Tm in Celsius using formula: Tm = 2*(A+T) + 4*(G+C)"
    },
    "gc_skew": {
        "description": "Calculate GC skew: (G-C)/(G+C) for a DNA sequence",
        "input": {"sequence": "DNA sequence"},
        "output": "GC skew value (-1 to +1)"
    },
}


def execute_llm_tool(tool_name: str, arguments: dict) -> ToolResult:
    """Execute an LLM-generated tool by having the LLM perform the computation."""
    if tool_name not in LLM_TOOLS:
        return ToolResult(
            success=False,
            result=None,
            explanation=f"Unknown tool: {tool_name}",
            confidence=0.0
        )

    tool_info = LLM_TOOLS[tool_name]
    return generate_and_execute_tool(tool_info['description'], arguments)


# Agent Response Models
class ToolCall(BaseModel):
    """A tool call with reasoning."""

    tool_name: str = Field(description="Name of tool to call")
    arguments: Dict[str, Any] = Field(description="Tool arguments as key-value pairs")
    reasoning: str = Field(description="Why this tool is being called")
    expected_outcome: str = Field(description="What information do you expect to get?")


class AgentReflection(BaseModel):
    """Agent's reflection on current state."""

    what_have_we_learned: str = Field(description="Summary of findings so far")
    what_still_unknown: str = Field(description="What questions remain unanswered")
    next_step_rationale: str = Field(description="Why the next action makes sense")
    confidence_in_answer: float = Field(description="Confidence 0-1 in having enough info", ge=0, le=1)
    ready_to_conclude: bool = Field(description="Whether we have enough info to answer")


class AgentResponse(BaseModel):
    """Agent's decision at each iteration."""

    reflection: Optional[AgentReflection] = Field(default=None, description="Agent's reflection on progress")
    tool_calls: List[ToolCall] = Field(description="Tools to call this iteration")
    final_answer: Optional[str] = Field(default=None, description="Final answer if task is complete")
    done: bool = Field(description="Whether the task is complete")


# LLM-Generated Tool Agent
# Why a class: same as demo_04 -- agent needs state (memory) across iterations.
#   Without conversation_history, the LLM would have no context of previous findings.
class LLMGeneratedToolAgent:
    """Agent that uses LLM-generated tools instead of hardcoded functions."""

    def __init__(self, max_iterations: int = 8, max_retries: int = 2):
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.conversation_history = []
        self.tool_results = []

        self.system_prompt = f"""You are a bioinformatics assistant with access to LLM-generated computational tools.

AVAILABLE LLM-GENERATED TOOLS:
{json.dumps(LLM_TOOLS, indent=2)}

HOW IT WORKS:
- Tools are NOT hardcoded - they are generated by LLM at runtime
- This is flexible but may be less accurate than hardcoded functions
- Always verify results make biological sense

WORKFLOW:
1. REFLECT on what you know and what you need
2. PLAN which LLM tools will help
3. EXECUTE ONE TOOL AT A TIME
4. VERIFY results make biological sense
5. SYNTHESIZE findings after all tools complete

# Why one tool at a time: LLM-generated tools are volatile. If you call three
#   in parallel and the second fails, the third's context may be wrong.
#   Sequential execution lets the agent adapt after each step.
CRITICAL GUIDELINES:
- Call ONE tool per iteration, then wait for results
- For multi-part tasks, do them one at a time
- Never make up data
"""

    def _execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute an LLM-generated tool with retry."""
        # Why retry: LLM calls are probabilistic -- sometimes they fail or
        #   return bad JSON. Retry is cheaper than giving up entirely.
        for attempt in range(self.max_retries):
            try:
                result = execute_llm_tool(tool_name, arguments)
                if result.success:
                    return result
                elif attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return ToolResult(
                    success=False,
                    result=None,
                    explanation=f"Tool execution failed: {str(e)}",
                    confidence=0.0
                )
        return ToolResult(success=False, result=None, explanation="Max retries", confidence=0.0)

    def run(self, task: str, verbose: bool = True) -> dict:
        """Run the agentic workflow with LLM-generated tools."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        self.tool_results = []

        if verbose:
            print(f"\nStarting LLM-generated tool workflow for: {task[:80]}...")
            print("=" * 60)
            print("NOTE: Tools are generated by LLM, not hardcoded!")
            print("=" * 60)

        response = None
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    response_model=AgentResponse,
                    messages=self.conversation_history
                )
            except Exception as e:
                if verbose:
                    print(f"  LLM call failed: {e}")
                continue

            if response.reflection and verbose:
                print(f"  Reflection: {response.reflection.what_still_unknown}...")
                print(f"  Confidence: {response.reflection.confidence_in_answer:.0%}")

            if response.done and response.final_answer:
                if verbose:
                    print(f"\nTask complete in {iteration + 1} iterations")
                    print(f"\n{'='*60}")
                    print("FINAL ANSWER:")
                    print(f"{'='*60}")
                    print(response.final_answer)

                return {
                    "success": True,
                    "final_answer": response.final_answer,
                    "tool_results": self.tool_results,
                    "iterations": iteration + 1,
                    "tool_type": "LLM-generated"
                }

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if verbose:
                        print(f"\n  LLM Tool: {tool_call.tool_name}")
                        print(f"     Args: {tool_call.arguments}")

                    result = self._execute_tool(tool_call.tool_name, tool_call.arguments)
                    self.tool_results.append({
                        "tool": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "result": result.model_dump() if hasattr(result, 'model_dump') else result,
                        "tool_type": "LLM-generated"
                    })

                    if verbose:
                        if result.success:
                            print("     OK")
                            print(f"     Explanation: {result.explanation}...")
                            print("     WARNING: LLMs pattern-match, not compute. Verify for science!")
                        else:
                            print(f"     Error: {result.explanation}")

                # Add to conversation history.
                # Why assistant+user pair: same chat protocol as demo_04.
                #   assistant message = what the agent decided.
                #   user message = what happened when tools ran.
                #   Next LLM call sees the full chain: plan -> result -> plan...
                self.conversation_history.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "reflection": response.reflection.model_dump() if response.reflection else None,
                        "tool_calls": [tc.model_dump() for tc in response.tool_calls]
                    })
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Tool results: {json.dumps(self.tool_results[-len(response.tool_calls):], default=str)}"
                })

        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "partial_answer": response.final_answer if response else None,
            "tool_results": self.tool_results,
            "iterations": self.max_iterations,
            "tool_type": "LLM-generated"
        }


def compare_approaches():
    """Show the difference between hardcoded and LLM-generated tools."""
    print("""
+------------------------------------------------------------------+
|           HARDCODED vs LLM-GENERATED TOOLS                       |
+------------------------------------------------------------------+
| Aspect          | Hardcoded (04)     | LLM-Generated (05)       |
+-----------------+--------------------+----------------------------+
| Speed           | Fast (Python)      | Slower (LLM call)          |
| Accuracy        | 100% deterministic | May hallucinate            |
| Flexibility     | Limited            | Any LLM-understood task    |
| Debugging       | Easy               | Hard                       |
| Best for        | Production         | Prototyping/exploration    |
+------------------------------------------------------------------+
""")


# Demo Tasks
DEMO_TASKS = [
    {
        "name": "Basic Sequence Analysis",
        "task": "Calculate the GC content of ATGCGATCGATCG and get its reverse complement.",
        "expected_tools": ["count_bases", "reverse_complement"]
    },
    {
        "name": "ORF Finding",
        "task": "Find ORFs in: ATGGCTGACTACGTGGCTAGCCTGCCAGGCCGCTGCTGGCATGCTGACTACGTGGCTAGCTAA",
        "expected_tools": ["find_orfs"]
    },
    {
        "name": "Multi-Step Analysis",
        "task": "Analyze this sequence: ATGGCTGACTACGTGGCTAGCCTGCCAGGCCGCTGCTGGCATGCTGACTACGTGGCTAGCTAA. Do: 1) Find ORFs, 2) Calculate GC content, 3) Get reverse complement",
        "expected_tools": ["find_orfs", "count_bases", "reverse_complement"]
    }
]


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 05: LLM-Generated Tools (No Hardcoded Logic)")
    print("=" * 60)
    print("\nThis demo shows LLM AS THE COMPUTATION:")
    print("  - Tools are NOT hardcoded Python functions")
    print("  - LLM generates tool logic at runtime")
    print()

    compare_approaches()

    # Run multi-step task
    selected = 3
    task_info = DEMO_TASKS[selected - 1]
    print(f"Running: {task_info['name']}")
    print(f"Task: {task_info['task']}")
    print()

    agent = LLMGeneratedToolAgent(max_iterations=8, max_retries=2)
    # Why max_iterations=8: LLM-generated tools are slower and less reliable
    #   than hardcoded ones (demo_04). More iterations gives the agent headroom
    #   to recover from errors or retry failed computations.
    result = agent.run(task_info['task'], verbose=True)

    # Summary
    print()
    print("=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Status: {'Success' if result['success'] else 'Incomplete'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools called: {len(result['tool_results'])}")
    print(f"Tool type: {result.get('tool_type', 'Unknown')}")

    print()
    print("=" * 60)
    print("CRITICAL WARNINGS:")
    print("  - LLMs pattern-match, they don't compute")
    print("  - Explanations are post-hoc rationalizations")
    print("  - NEVER use for clinical, diagnostic, or research work")
    print("  - ALWAYS verify with hardcoded tools (demo_04)")
    print("  - Use ONLY for: prototyping, education, exploration")
    print("=" * 60)
