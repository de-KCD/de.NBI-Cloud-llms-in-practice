"""
demo_05_llm_limitations.py - Understanding LLM Limitations (Pattern-Matching, NOT Computation)

⚠️  CRITICAL WARNING: THIS DEMO IS FOR EDUCATION ONLY! 🤯

This demo shows what LLMs APPEAR to do, but remember:
LLMs DON'T actually calculate - they PATTERN-MATCH from training data!

THE KEY INSIGHT:

❌ What people think LLMs do (demo_04 - Real API Agent):

   Human writes: def count_bases(seq):
                    # hardcoded logic
                    return result

   Agent calls: count_bases("ATGC")

   → 100% accurate, deterministic, reproducible

✅ What LLMs ACTUALLY do (this demo):

   Human describes: "Calculate GC content"

   LLM generates text that LOOKS LIKE: "GC% = 50%"

   → Pattern-matching from training data, NOT real calculation!
   → For short sequences, LLM has seen many examples → looks correct
   → For long/complex sequences → LLM guesses → errors!

WHAT THIS DEMONSTRATES:

1. LLM AS PATTERN-MATCHER (not computation)
   - LLM doesn't "calculate" - it generates plausible text
   - "Explanations" are post-hoc rationalizations, not actual work
   - Confidence scores are HALLUCINATED (just another generated token)

2. WHY THIS IS DANGEROUS FOR SCIENCE:
   ✗ Arithmetic errors common (especially long sequences)
   ✗ Confidence ≠ accuracy (LLM can be confidently wrong)
   ✗ Not reproducible (LLM updates change behavior)
   ✗ NEVER use for clinical/diagnostic/production work

3. WHEN LLMs DO WELL:
   ✓ Short sequences (<50 bases)
   ✓ Common patterns (GC%, reverse complement)
   ✓ Tasks seen frequently in training

4. WHEN LLMs FAIL:
   ✗ Long sequences (>100 bases) - lose track
   ✗ Novel calculations - no training patterns
   ✗ Multi-step arithmetic - compounding errors

COMPARISON TABLE:

┌─────────────────────┬────────────────────┬─────────────────────────┐
│ Aspect              │ Hardcoded (04)     │ LLM Pattern-Match (05)  │
├─────────────────────┼────────────────────┼─────────────────────────┤
│ Speed               │ Fast (native Python)│ Slower (LLM API call)  │
│ Accuracy            │ 100% deterministic │ ~85-95% (varies!)       │
│ Confidence          │ N/A (deterministic)│ HALLUCINATED ⚠️        │
│ Flexibility         │ Limited to defined │ Any LLM-understood task │
│ Best for            │ Production, science│ Prototyping, education  │
└─────────────────────┴────────────────────┴─────────────────────────┘

GOLDEN RULE:

If it matters scientifically → VERIFY with hardcoded tools (demo_04)!
This demo is for EDUCATION ONLY - to understand LLM limitations!

"""

# =============================================================================
# IMPORTS
# =============================================================================
import json
import time
import instructor
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================
# API credentials and settings
#
# ⚠️ SECURITY NOTE: In production, use environment variables!

API_KEY = "your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 8192
TIMEOUT = 300.0

# Setup client with JSON mode for structured outputs
client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# =============================================================================
# LLM-GENERATED TOOL SYSTEM
# =============================================================================
# This is the REVOLUTIONARY part!
#
# Instead of:
#   TOOLS = {"count_bases": count_bases_function, ...}
#
# We have:
#   LLM_TOOLS = {"count_bases": {"description": "...", "input": {...}}, ...}
#
# The LLM itself performs the computation when a tool is called!
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TOOL RESULT MODEL
# ─────────────────────────────────────────────────────────────────────────────
# This model structures the output from LLM-generated tools.
#
# Key fields:
# - success: Did the computation work?
# - result: The actual computed value
# - explanation: HOW the LLM computed it (crucial for debugging!)
# - confidence: How sure is the LLM? (0-1)
# ─────────────────────────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """
    Result from an LLM-generated tool.

    Unlike hardcoded tools that return simple dicts, LLM-generated
    tools return structured results with explanations.

    ⚠️  IMPORTANT: The LLM does NOT actually calculate anything!

    LLMs are language models, not calculators. They generate text that
    LOOKS LIKE calculations based on training patterns. The "explanation"
    is plausible-sounding reasoning, not actual computational steps.

    For simple, common patterns (GC% of short sequences), the LLM has
    seen enough examples to generate correct-looking answers. But it's
    pattern matching, not computing.

    ⚠️  NEVER trust LLM "calculations" for real science!
    - Confidence is hallucinated (just another generated token)
    - Explanations are post-hoc rationalizations, not actual work
    - Arithmetic errors are common, especially with longer sequences
    - Always verify with hardcoded tools (demo_04) for real work

    Why explanations are still useful:
    - You can spot-check the reasoning ("Wait, that formula is wrong!")
    - Helps understand what the LLM was attempting
    - Educational: shows common misconceptions

    Example:
        ToolResult(
            success=True,
            result=50.0,  # ⚠️  May be wrong for complex/long inputs
            explanation="Counted G=2, C=2... GC% = 50%",  # ⚠️  Not real work!
            confidence=0.95  # ⚠️  Hallucinated - just generated text!
        )
    """
    success: bool = Field(description="Whether the tool succeeded")
    result: Any = Field(description="The computed result - VERIFY for science!")
    explanation: str = Field(
        description="Plausible reasoning (NOT actual computation steps - LLMs don't calculate!)"
    )
    confidence: float = Field(
        description="LLM self-assessment (0-1) - NOT reliable! Just generated text.",
        ge=0,
        le=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATED TOOL MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Metadata about each LLM-generated tool.
#
# This is the "interface definition" - tells the agent:
# - What the tool does
# - What input it expects
# - What output it produces
# ─────────────────────────────────────────────────────────────────────────────

class GeneratedTool(BaseModel):
    """Metadata about an LLM-generated tool."""
    tool_name: str = Field(description="Name of the generated tool")
    description: str = Field(description="What this tool does")
    input_description: str = Field(description="Expected input format")
    output_description: str = Field(description="Expected output format")


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION: GENERATE AND EXECUTE TOOL
# ─────────────────────────────────────────────────────────────────────────────
# This is the MAGIC! Instead of calling a Python function, we:
# 1. Send the task description to the LLM
# 2. LLM figures out HOW to compute it
# 3. LLM returns the result + explanation
#
# The LLM IS the computation engine!
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_execute_tool(task_description: str, input_data: Any) -> ToolResult:
    """
    Generate a tool on-the-fly and execute it.

    This function embodies "LLM as computation":

    Traditional approach:
        def count_bases(seq):
            # Human wrote this logic
            counts = {b: seq.count(b) for b in 'ATGC'}
            return counts

    LLM approach:
        LLM prompt: "Count bases in ATGC"
        LLM thinks: "A=1, T=1, G=1, C=1. Total=4"
        LLM returns: {"A": 1, "T": 1, "G": 1, "C": 1}

    Args:
        task_description: What computation to perform
                         Example: "Calculate GC content"
        input_data: The data to compute on
                   Example: {"sequence": "ATGCATGC"}

    Returns:
        ToolResult with:
        - success: True/False
        - result: The computed value
        - explanation: Step-by-step reasoning
        - confidence: LLM's confidence (0-1)

    Why this is powerful:
    - No coding needed - just describe what you want
    - Works for ANY computation the LLM understands
    - LLM can adapt to edge cases
    - Explanation shows the reasoning

    Why this is risky:
    - LLMs can make arithmetic errors
    - May hallucinate methods or formulas
    - Slower than native Python
    - Harder to verify correctness
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            response_model=ToolResult,
            messages=[
                {
                    "role": "system",
                    "content": """You are a computational assistant. Given a task description and input data,
perform the computation and return the result.

⚠️  HONESTY REQUIREMENT:
- LLMs don't actually calculate - you generate text that looks like calculations
- Be honest about uncertainty in your explanation
- If you're pattern-matching from training, say so
- If the sequence is long or calculation is complex, warn the user
- NEVER make up formulas or methods you're unsure about

GUIDELINES:
- For simple, common tasks (GC% of short sequences), you've seen many examples
- For complex/long inputs, warn that verification is needed
- Show your reasoning, but acknowledge it's generated text, not actual computation
- If unsure, recommend using hardcoded tools (Python functions) instead

EXAMPLES:

Task: "Calculate GC content"
Input: {"sequence": "ATGCATGC"}
Honest approach:
  "This is a common pattern I've seen. For ATGCATGC:
   - Length = 8 bases
   - G count = 2, C count = 2
   - GC% = (2+2)/8 × 100 = 50%
   Note: For longer sequences, verify with a calculator!"
Result: 50.0
"""
                },
                {
                    "role": "user",
                    "content": f"""Task: {task_description}
Input: {json.dumps(input_data, default=str)}

Perform this computation and return the result.
SHOW YOUR WORK in the explanation field."""
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


# =============================================================================
# LLM-GENERATED TOOL REGISTRY
# =============================================================================
# Instead of hardcoded functions, we have DESCRIPTIONS.
#
# Each tool entry has:
# - description: What it does
# - input: What data it needs
# - output: What it returns
#
# The LLM reads these descriptions and figures out the implementation!
# =============================================================================

LLM_TOOLS = {
    # ──────────────────────────────────────────────────────────────────────────
    # BASIC SEQUENCE TOOLS
    # ──────────────────────────────────────────────────────────────────────────

    "reverse_complement": {
        "description": "Get the reverse complement of a DNA sequence",
        "input": {"sequence": "DNA sequence string (e.g., 'ATGCATGC')"},
        "output": "Reverse complement sequence (e.g., 'ATGC' → 'GCAT')"
    },

    "count_bases": {
        "description": "Count each base (A,T,G,C) in a DNA sequence and calculate GC content",
        "input": {"sequence": "DNA sequence string"},
        "output": "Dict with counts {'A': n, 'T': n, 'G': n, 'C': n} and gc_content percentage"
    },

    "translate_dna": {
        "description": "Translate DNA sequence to protein using standard genetic code",
        "input": {
            "sequence": "DNA sequence",
            "frame": "Reading frame (1, 2, or 3)"
        },
        "output": "Protein sequence (single letter amino acid codes, e.g., 'MADYV')"
    },

    "find_orfs": {
        "description": "Find open reading frames in DNA sequence",
        "input": {
            "sequence": "DNA sequence",
            "min_length": "Minimum ORF length in nucleotides (default: 30)"
        },
        "output": "List of ORFs with start position, end position, frame, length"
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ADVANCED SEQUENCE TOOLS
    # ──────────────────────────────────────────────────────────────────────────

    "calculate_melting_temp": {
        "description": "Calculate melting temperature (Tm) of DNA sequence using Wallace rule",
        "input": {"sequence": "DNA sequence"},
        "output": "Tm in Celsius using formula: Tm = 2×(A+T) + 4×(G+C)"
    },

    "gc_skew": {
        "description": "Calculate GC skew: (G-C)/(G+C) for a DNA sequence",
        "input": {"sequence": "DNA sequence"},
        "output": "GC skew value (-1 to +1); positive = G-rich, negative = C-rich"
    },

    "reverse_sequence": {
        "description": "Reverse a DNA sequence (not complement, just reverse)",
        "input": {"sequence": "DNA sequence"},
        "output": "Reversed sequence (e.g., 'ATGC' → 'CGTA')"
    },

    "complement_sequence": {
        "description": "Get complement of DNA sequence (A↔T, G↔C) without reversing",
        "input": {"sequence": "DNA sequence"},
        "output": "Complement sequence (e.g., 'ATGC' → 'TACG')"
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────
# This function bridges the agent's tool call to the LLM computation.
#
# Flow:
# 1. Agent says: "Call count_bases with sequence='ATGC'"
# 2. We look up the tool description from LLM_TOOLS
# 3. We call generate_and_execute_tool() with the description
# 4. LLM performs the computation and returns result
# ─────────────────────────────────────────────────────────────────────────────

def execute_llm_tool(tool_name: str, arguments: dict) -> ToolResult:
    """
    Execute an LLM-generated tool.

    This is the "dispatcher" - it routes tool calls to the LLM.

    Args:
        tool_name: Name of tool to call (must match LLM_TOOLS keys)
        arguments: Dict of arguments to pass

    Returns:
        ToolResult with computation result

    How it works:
    1. Look up tool description from LLM_TOOLS
    2. Build task description for LLM
    3. Call generate_and_execute_tool()
    4. Return LLM's result

    Example:
        >>> result = execute_llm_tool("count_bases", {"sequence": "ATGCATGC"})
        >>> print(f"GC content: {result.result['gc_content']}%")
        >>> print(f"How: {result.explanation}")
    """
    # Check if tool exists
    if tool_name not in LLM_TOOLS:
        return ToolResult(
            success=False,
            result=None,
            explanation=f"Unknown tool: {tool_name}. Available: {list(LLM_TOOLS.keys())}",
            confidence=0.0
        )

    # Get tool description
    tool_info = LLM_TOOLS[tool_name]

    # Build task description for LLM
    task = f"{tool_info['description']}"

    # Call LLM to perform the computation
    return generate_and_execute_tool(task, arguments)


# =============================================================================
# AGENT RESPONSE MODELS
# =============================================================================
# Same models as demo_04 - the agent structure is identical
# Only the tool execution mechanism differs
# =============================================================================

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


# =============================================================================
# LLM-GENERATED TOOL AGENT
# =============================================================================
# This agent is nearly identical to AgenticWorkflow from demo_04!
# The ONLY difference is HOW tools are executed:
# - demo_04: TOOLS[tool_name](**arguments) - Python function
# - demo_05: execute_llm_tool(tool_name, arguments) - LLM computation
# =============================================================================

class LLMGeneratedToolAgent:
    """
    Agent that uses LLM-generated tools instead of hardcoded functions.

    This agent demonstrates a fundamentally different approach to tool-calling:

    TRADITIONAL (demo_04):
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Agent      │────▶│  TOOLS dict  │────▶│  Python func │
    │  decides     │     │  (hardcoded) │     │  (fixed)     │
    └──────────────┘     └──────────────┘     └──────────────┘

    LLM-GENERATED (demo_05):
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Agent      │────▶│  LLM_TOOLS   │────▶│     LLM      │
    │  decides     │     │ (descriptions)│    │ (computation)│
    └──────────────┘     └──────────────┘     └──────────────┘

    Key differences:
    - Tools are DESCRIPTIONS, not functions
    - LLM performs the COMPUTATION, not just orchestration
    - Infinite flexibility (any computation LLM understands)
    - Trade-off: slower and less reliable

    Use cases:
    ✓ Rapid prototyping
    ✓ Educational demos
    ✓ Exploratory analysis
    ✗ Production systems
    ✗ Critical applications
    ✗ High-throughput processing
    """

    def __init__(self, max_iterations: int = 8, max_retries: int = 2):
        """
        Initialize the LLM-generated tool agent.

        Args:
            max_iterations: Maximum tool-calling rounds (default: 8)
            max_retries: How many times to retry failed tool calls
        """
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.conversation_history = []
        self.tool_results = []

        # System prompt explains the LLM-generated tool concept to the agent
        self.system_prompt = f"""You are a bioinformatics assistant with access to LLM-generated computational tools.

AVAILABLE LLM-GENERATED TOOLS:
{json.dumps(LLM_TOOLS, indent=2)}

HOW IT WORKS:
- Tools are NOT hardcoded - they are generated by LLM at runtime
- Describe what you want, the LLM figures out how to compute it
- This is flexible but may be less accurate than hardcoded functions
- Always verify results make biological sense
- Pay attention to the 'explanation' field - it shows the LLM's reasoning

WORKFLOW:
1. REFLECT on what you know and what you need
2. PLAN which LLM tools will help
3. EXECUTE **ONE TOOL AT A TIME** - don't call multiple tools in one iteration
4. VERIFY results make biological sense
5. SYNTHESIZE findings after all tools complete

CRITICAL GUIDELINES:
- Call ONE tool per iteration, then wait for results
- For multi-part tasks (e.g., "1) X, 2) Y, 3) Z"), do them one at a time
- After each tool result, re-evaluate before calling the next
- Only set done=True when ALL subtasks are complete
- If a tool gives weird results, try again or use a different approach
- Never make up data - if you can't compute it, say so

EXAMPLE MULTI-STEP APPROACH:
Task: "Calculate GC content and get reverse complement"
- Iteration 1: Call count_bases → wait for result
- Iteration 2: Call reverse_complement → wait for result
- Iteration 3: Synthesize both results → done=True
"""

    def _execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """
        Execute an LLM-generated tool with retry.

        This wraps execute_llm_tool() with error handling.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            ToolResult with computation result
        """
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
        """
        Run the agentic workflow with LLM-generated tools.

        This is nearly identical to AgenticWorkflow.run() from demo_04.
        The only difference is the tool execution mechanism.

        Args:
            task: The user's question/request
            verbose: Whether to print progress

        Returns:
            dict with final_answer, tool_results, iterations
        """
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        self.tool_results = []

        if verbose:
            print(f"\n🔬 Starting LLM-generated tool workflow for: {task[:80]}...")
            print("=" * 60)
            print("NOTE: Tools are generated by LLM, not hardcoded!")
            print("=" * 60)

        response = None
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Get agent's decision
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    response_model=AgentResponse,
                    messages=self.conversation_history
                )
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  LLM call failed: {e}")
                continue

            # Log reflection
            if response.reflection and verbose:
                print(f"  🤔 Reflection: {response.reflection.what_still_unknown}...")
                print(f"  📊 Confidence: {response.reflection.confidence_in_answer:.0%}")

            # Check if done
            if response.done and response.final_answer:
                if verbose:
                    print(f"\n✅ Task complete in {iteration + 1} iterations")
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

            # Execute tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if verbose:
                        print(f"\n  🔧 LLM Tool: {tool_call.tool_name}")
                        print(f"     Args: {tool_call.arguments}")
                        print(f"     Why: {tool_call.reasoning}...")

                    result = self._execute_tool(tool_call.tool_name, tool_call.arguments)
                    self.tool_results.append({
                        "tool": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "result": result.model_dump() if hasattr(result, 'model_dump') else result,
                        "tool_type": "LLM-generated"
                    })

                    if verbose:
                        if result.success:
                            print("     ✓ Success")
                            print(f"     Explanation: {result.explanation}...")
                            print("     ⚠️  LLMs don't calculate - this is pattern-matching, not computation!")
                            print("     ⚠️  Verify with demo_04 for real science!")
                        else:
                            print(f"     ✗ Error: {result.explanation}")

                # Add to conversation history
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

        # Max iterations reached
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "partial_answer": response.final_answer if response else None,
            "tool_results": self.tool_results,
            "iterations": self.max_iterations,
            "tool_type": "LLM-generated"
        }


# =============================================================================
# DEMO TASKS
# =============================================================================
# Realistic DNA sequences for meaningful bioinformatics analysis
# These tasks showcase what LLM-generated tools can handle
# =============================================================================

DEMO_TASKS = [
    {
        "name": "Basic Sequence Analysis",
        "task": "Calculate the GC content of ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC and get its reverse complement.",
        "expected_tools": ["count_bases", "reverse_complement"]
    },
    {
        "name": "ORF Finding",
        "task": "Find ORFs in this sequence: ATGGCTGACTACGTAGCTAGCTAGCTAGCTAGCTAG",
        "expected_tools": ["find_orfs"]
    },
    {
        "name": "Translation",
        "task": "Translate this DNA to protein: ATGGCTGACTACGTAGCTAGCTAGCTAGCTAGCTAG",
        "expected_tools": ["translate_dna"]
    },
    {
        "name": "Multi-Step Analysis",
        "task": "Analyze this promoter region sequence: ATGGCTGACTACGTAGCTAGCTAGCTAGCTAGCTAG. Do: 1) Find ORFs, 2) Calculate GC content, 3) Get reverse complement, 4) Calculate melting temperature, 5) Calculate GC skew",
        "expected_tools": ["find_orfs", "count_bases", "reverse_complement", "calculate_melting_temp", "gc_skew"]
    }
]


# =============================================================================
# COMPARISON: HARDCODED VS LLM-GENERATED
# =============================================================================
# This function prints a comparison table showing the trade-offs
# =============================================================================

def compare_approaches():
    """Show the difference between hardcoded and LLM-generated tools."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   HARDCODED vs LLM-GENERATED TOOLS                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Aspect              │ Hardcoded (04)      │ LLM-Generated (05)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Speed               │ Fast (native Python)│ Slower (LLM call per tool)      ║
║  Accuracy            │ 100% deterministic  │ May hallucinate/miscalculate    ║
║  Flexibility         │ Limited to defined  │ Any computation LLM understands ║
║  Maintenance         │ Update code         │ Update descriptions             ║
║  Debugging           │ Easy to trace       │ Hard to debug LLM reasoning     ║
║  Best for            │ Production, critical│ Prototyping, exploration        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# DEMO MAIN
# =============================================================================
# Let's see LLM-generated tools in action!
#
# This demo shows:
# 1. The comparison table (hardcoded vs LLM-generated)
# 2. A multi-step analysis task
# 3. Tool execution trace with confidence scores
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 05: LLM-Generated Tools (No Hardcoded Logic)")
    print("=" * 60)
    print()
    print("This demo shows LLM AS THE COMPUTATION:")
    print("  • Tools are NOT hardcoded Python functions")
    print("  • LLM generates tool logic at runtime")
    print("  • Describe what you want, LLM figures out how")
    print()

    # Show comparison table
    compare_approaches()

    # Run selected demo task
    print("Select demo task:")
    for i, task_info in enumerate(DEMO_TASKS, 1):
        print(f"  {i}. {task_info['name']}")
    print()

    # Default to task 4 (multi-step) for full demo
    selected = 4
    task_info = DEMO_TASKS[selected - 1]

    print(f"Running: {task_info['name']}")
    print(f"Task: {task_info['task']}")
    print()

    # Create and run agent
    agent = LLMGeneratedToolAgent(max_iterations=8, max_retries=2)
    result = agent.run(task_info['task'], verbose=True)

    # Summary
    print()
    print("=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Status: {'✓ Success' if result['success'] else '✗ Incomplete'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools called: {len(result['tool_results'])}")
    print(f"Tool type: {result.get('tool_type', 'Unknown')}")

    if result['tool_results']:
        print("\nTool execution trace:")
        for i, tr in enumerate(result['tool_results'], 1):
            status = "✓" if tr['result'].get('success') else "✗"
            conf = tr['result'].get('confidence', 'N/A')
            print(f"  {i}. {status} {tr['tool']} (confidence: {conf})")

    print()
    print("=" * 60)
    print("DEMO 05 COMPLETE")
    print("=" * 60)
    print()
    print("What you saw:")
    print("  ✓ LLM as pattern-matcher (not computation)")
    print("  ✓ Tools are descriptions, not hardcoded functions")
    print("  ✓ Generated explanations (plausible reasoning, not actual work)")
    print("  ⚠️  LLMs don't calculate - they generate text that looks like math!")
    print("  ⚠️  Confidence and explanations are NOT reliable for science!")
    print("  ✓ Trade-offs: flexibility vs accuracy")
    print()
    print("Critical warnings:")
    print("  → LLMs pattern-match, they don't compute")
    print("  → Explanations are post-hoc rationalizations")
    print("  → NEVER use for clinical, diagnostic, or research work")
    print("  → ALWAYS verify with hardcoded tools (demo_04)")
    print("  → Use ONLY for: prototyping, education, exploration")
    print()
    print("Try it yourself:")
    print("  → Add new tool descriptions to LLM_TOOLS dict")
    print("  → Test with your own DNA sequences")
    print("  → Compare results with demo_04 (hardcoded tools)")
    print()
    print("Available LLM tools:")
    print("  • reverse_complement, count_bases, translate_dna")
    print("  • find_orfs, calculate_melting_temp, gc_skew")
    print("  • reverse_sequence, complement_sequence")
    print()
    print("Next steps:")
    print("  → demo_06: Fully autonomous investigation")
    print("  → Compare: When to use hardcoded vs LLM-generated")
    print("=" * 60)
