"""
demo_04_real_api_agent.py - Real Agentic Workflow for Bioinformatics

Agent with REAL API tools connecting to actual bioinformatics databases:
- UniProtClient: Protein database queries
- LiteratureClient: PubMed literature search
- SequenceAnalysisClient: DNA/protein sequence analysis

Key features:
- Reflection before acting (what do I know? what do I need?)
- Multi-step reasoning workflows
- Error handling and retry logic
- Conversation memory across iterations

"""

import json
import time
import requests
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
# Why: CoT is great for exploration, but we're using structured outputs.
#   The model's "thinking" would leak into the JSON fields as noise.
EXTRA_BODY = {
    "chat_template_kwargs": {
        "enable_thinking": False,
        "preserve_thinking": False
    }
}

# Instructor wraps OpenAI client -> enforces Pydantic schema on LLM output.
# Why JSON mode: guarantees parseable output. Without it, the LLM returns
#   free text and you'd need regex/parsing to extract structured data.
client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# UniProt Client
# Why a separate class: each external API gets its own client so tools
#   remain testable, swappable, and isolated from LLM call logic.
class UniProtClient:
    """Client for UniProt protein database (https://uniprot.org)."""

    BASE_URL = "https://rest.uniprot.org"

    def search_protein(self, gene_name: str, organism: str = "human") -> dict:
        """Search UniProt for a protein by gene name."""
        try:
            query = f"gene:{gene_name} AND reviewed:true"
            if organism.lower() in ["human", "homo sapiens", "9606"]:
                query += " AND organism_id:9606"

            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/search",
                params={"query": query, "format": "json", "size": 1},
                timeout=TIMEOUT
            )

            if resp.status_code == 400 or (resp.ok and not resp.json().get("results")):
                # Fallback without organism filter or reviewed filter.
                # Why: gene names like "BRCA1" may exist in multiple species.
                #   If the user says "human" but UniProt's query syntax rejects it,
                #   we still want the result rather than failing outright.
                resp = requests.get(
                    f"{self.BASE_URL}/uniprotkb/search",
                    params={"query": f"gene:{gene_name}", "format": "json", "size": 1},
                    timeout=TIMEOUT
                )

            resp.raise_for_status()
            data = resp.json()

            if not data.get("results"):
                return {"success": False, "error": f"Protein '{gene_name}' not found"}

            result = data["results"][0]
            return {
                "success": True,
                "uniprot_id": result.get("primaryAccession"),
                "entry_name": result.get("uniProtkbId"),
                "protein_name": result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value"),
                "gene_name": gene_name,
                "organism": result.get("organism", {}).get("scientificName"),
                "length": result.get("sequence", {}).get("length")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_protein_function(self, uniprot_id: str) -> dict:
        """Get protein function annotations from UniProt."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/{uniprot_id}",
                params={"format": "json"},
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            functions = []
            for comment in data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    for text in texts:
                        value = text.get("value", "")
                        if value:
                            functions.append(value)

            return {
                "success": True,
                "uniprot_id": uniprot_id,
                "functions": functions if functions else ["No function annotation available"],
                "protein_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Literature Client
# Why a separate class: PubMed has its own auth, rate limits, and quirks.
#   Isolating it means LLM logic doesn't break if PubMed changes.
class LiteratureClient:
    """Client for biomedical literature search via PubMed (NCBI E-utilities)."""

    def search_pubmed(self, query: str, max_results: int = 3) -> dict:
        """Search PubMed for relevant papers."""
        try:
            # PubMed E-utilities is two-step: search then fetch.
            # Why not one call: NCBI's API is designed this way for performance.
            #   ESEARCH returns PMIDs (cheap, fast). ESUMMARY fetches details.
            #   This pattern lets you paginate, filter, or deduplicate before
            #   making the heavier fetch calls.
            # Step 1: ESEARCH - find PMIDs
            search_resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
                timeout=TIMEOUT
            )
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return {"success": False, "error": "No papers found"}

            # Step 2: ESUMMARY - fetch details
            fetch_resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                timeout=TIMEOUT
            )
            fetch_resp.raise_for_status()

            papers = []
            for pmid in pmids[:max_results]:
                paper = fetch_resp.json().get("result", {}).get(pmid, {})
                authors_list = paper.get("authors", [])
                if isinstance(authors_list, list) and len(authors_list) > 0:
                    author_names = [a.get("name", str(a)) for a in authors_list[:3]]
                    authors_str = ", ".join(author_names) + " et al."
                else:
                    authors_str = "Unknown"

                papers.append({
                    "pmid": pmid,
                    "title": paper.get("title", ""),
                    "authors": authors_str,
                    "journal": paper.get("fulljournalname", ""),
                    "year": paper.get("pubdate", "")[:4]
                })

            return {"success": True, "query": query, "papers": papers}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Sequence Analysis Client
# Why local computation (not LLM): sequence analysis is deterministic math.
#   The LLM can't be trusted for codon translation -- it hallucinates.
#   Hardcoded genetic code table = 100% correct, zero latency.
class SequenceAnalysisClient:
    """DNA/protein sequence analysis tools (local computation)."""

    def translate_dna(self, sequence: str, frame: int = 1) -> dict:
        """Translate DNA sequence to protein using the genetic code."""
        # Standard genetic code table (64 codons -> 20 amino acids + 3 stops).
        # Why hardcoded: this is a biological constant. No need to fetch or
        #   compute it -- it's the same across nearly all life on Earth.
        genetic_code = {
            'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
            'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
            'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
            'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
            'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
            'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
            'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
            'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
            'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
            'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
        }

        try:
            seq = sequence.upper().replace(" ", "").replace("\n", "")
            # Frame 1 = start at position 0, frame 2 = skip 1 base, etc.
            # Why frame-1: Python uses 0-indexed slicing, but biologists number
            #   reading frames starting at 1.
            seq = seq[frame-1:] if frame > 0 else seq

            protein = []
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                aa = genetic_code.get(codon, 'X')
                protein.append(aa)
                if aa == '*':
                    break

            return {
                "success": True,
                "dna_length": len(seq),
                "protein_length": len(protein),
                "protein_sequence": ''.join(protein),
                "frame": frame,
                "has_stop": '*' in ''.join(protein)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def find_orfs(self, sequence: str, min_length: int = 30) -> dict:
        """Find open reading frames (ORFs) in DNA sequence."""
        # ORF = stretch from start codon (ATG) to stop codon (TAA/TAG/TGA).
        # Why scan all 3 frames: you don't know which frame is the correct one
        #   without additional annotation. The real gene could be in any frame.
        try:
            seq = sequence.upper().replace(" ", "").replace("\n", "")
            orfs = []
            start_codon = "ATG"
            stop_codons = ["TAA", "TAG", "TGA"]

            for frame in range(3):
                i = frame
                while i < len(seq) - 2:
                    if seq[i:i+3] == start_codon:
                        for j in range(i+3, len(seq)-2, 3):
                            if seq[j:j+3] in stop_codons:
                                orf_len = j + 3 - i
                                if orf_len >= min_length:
                                    orfs.append({
                                        "frame": frame + 1,
                                        "start": i,
                                        "end": j + 3,
                                        "length": orf_len,
                                        "protein_len": orf_len // 3
                                    })
                                break
                    i += 3

            return {
                "success": True,
                "sequence_length": len(seq),
                "orf_count": len(orfs),
                "longest_orf": max(orfs, key=lambda x: x["length"]) if orfs else None,
                "orfs": orfs[:5]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize clients
uniprot = UniProtClient()
literature = LiteratureClient()
seq_analysis = SequenceAnalysisClient()

# Tool Registry
# Why a dict of lambdas: gives the agent a flat namespace it can call by name.
#   The LLM outputs {"tool_name": "X", "arguments": {...}} and we just
#   dispatch with TOOLS[name](**args). Simple lookup, easy to extend.
TOOLS = {
    "search_uniprot": lambda gene, organism="human": uniprot.search_protein(gene, organism),
    "get_protein_function": lambda uniprot_id: uniprot.get_protein_function(uniprot_id),
    "search_literature": lambda query, max_results=3: literature.search_pubmed(query, max_results),
    "translate_dna": lambda sequence, frame=1: seq_analysis.translate_dna(sequence, frame),
    "find_orfs": lambda sequence, min_length=30: seq_analysis.find_orfs(sequence, min_length),
    "reverse_complement": lambda seq: {"success": True, "result": ''.join({'A':'T','T':'A','G':'C','C':'G'}[b] for b in reversed(seq.upper()))},
    "count_bases": lambda seq: {"success": True, "gc_content": round((seq.upper().count('G') + seq.upper().count('C')) / len(seq) * 100, 2)},
}

TOOL_DESCRIPTIONS = """
Available tools:
- search_uniprot(gene: str, organism: str) - Search UniProt protein database
- get_protein_function(uniprot_id: str) - Get protein function annotations
- search_literature(query: str, max_results: int) - Search PubMed literature
- translate_dna(sequence: str, frame: int) - Translate DNA to protein
- find_orfs(sequence: str, min_length: int) - Find open reading frames
- reverse_complement(sequence: str) - Get reverse complement of DNA
- count_bases(sequence: str) - Calculate GC content
"""


# Agent Response Models
# Why Pydantic: schema acts as a contract. The LLM must produce JSON matching
#   these fields, or instructor retries automatically. No manual parsing needed.
# Why separate ToolCall + Reflection models: the agent thinks (reflection)
#   before it acts (tool_calls). This forces deliberate multi-step reasoning.
class ToolCall(BaseModel):
    """A tool call with reasoning."""

    tool_name: str = Field(description="Name of tool to call")
    arguments: Dict[str, Any] = Field(description="Tool arguments as key-value pairs")
    reasoning: str = Field(description="Why this tool is being called - what question does it answer?")
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


# Agentic Workflow Class
# Why a class: the agent needs state across iterations (memory).
#   conversation_history = what the LLM sees. tool_results = execution trace.
#   Without this, every LLM call would be stateless and the agent couldn't
#   build on previous findings.
class AgenticWorkflow:
    """Real agentic workflow with reflection and multi-step reasoning."""

    def __init__(self, max_iterations: int = 8, max_retries: int = 2):
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.conversation_history = []
        self.tool_results = []

        self.system_prompt = f"""You are a bioinformatics research assistant with access to real databases and analysis tools.

{TOOL_DESCRIPTIONS}

WORKFLOW:
1. REFLECT on what you know and what you need to find out
2. PLAN which tools will answer remaining questions
3. EXECUTE tool calls in parallel when possible
4. SYNTHESIZE findings into a coherent answer

GUIDELINES:
- Always explain your reasoning for tool calls
- Use specific gene names, IDs, and sequences when available
- If a tool fails, try a different approach
- Cite literature with PMIDs when relevant
- Admit uncertainty rather than making up data
"""

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool with error handling and retry logic."""
        # Why retry: network calls to UniProt/PubMed are flaky.
        #   Linear backoff (1s sleep) is simpler than exponential and
        #   sufficient for our scale -- we're doing a few calls, not millions.
        if tool_name not in TOOLS:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        for attempt in range(self.max_retries):
            try:
                result = TOOLS[tool_name](**arguments)
                if isinstance(result, dict) and result.get("success"):
                    return result
                elif attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return {"success": False, "error": f"Tool execution failed: {str(e)}"}

        return {"success": False, "error": "Max retries exceeded"}

    def run(self, task: str, verbose: bool = True) -> dict:
        """Run the agentic workflow to complete a task."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        self.tool_results = []

        if verbose:
            print(f"\nStarting agentic workflow for: {task}...")
            print("=" * 60)

        response = None
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Step 1: Get agent's decision from LLM
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

            # Step 2: Log reflection
            if response.reflection and verbose:
                print(f"  Reflection: {response.reflection.what_still_unknown}...")
                print(f"  Confidence: {response.reflection.confidence_in_answer:.0%}")

            # Step 3: Check if done
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
                    "iterations": iteration + 1
                }

            # Step 4: Execute tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if verbose:
                        print(f"\n  Calling: {tool_call.tool_name}")
                        print(f"    Reason: {tool_call.reasoning}")

                    result = self._execute_tool(tool_call.tool_name, tool_call.arguments)
                    self.tool_results.append({
                        "tool": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "result": result
                    })

                    if verbose:
                        status = "Success" if result.get("success") else "Failed"
                        print(f"    Result: {status}")
                        if result.get("success"):
                            # Print key fields depending on tool type
                            if tool_call.tool_name == "search_uniprot":
                                print(f"      -> {result.get('protein_name')} (UniProt: {result.get('uniprot_id')})")
                            elif tool_call.tool_name == "get_protein_function":
                                for fn in result.get("functions", [])[:2]:
                                    print(f"      -> Function: {fn[:120]}...")
                            elif tool_call.tool_name == "search_literature":
                                for p in result.get("papers", [])[:2]:
                                    print(f"      -> [{p.get('year')}] {p.get('title')[:100]}...")
                            elif tool_call.tool_name == "translate_dna":
                                print(f"      -> Protein ({result.get('protein_length')} aa): {result.get('protein_sequence', '')[:60]}...")
                            elif tool_call.tool_name == "find_orfs":
                                print(f"      -> Found {result.get('orf_count')} ORFs (longest: {result.get('longest_orf', {}).get('protein_len')} aa)")
                            elif tool_call.tool_name == "count_bases":
                                print(f"      -> GC content: {result.get('gc_content')}%")
                            elif tool_call.tool_name == "reverse_complement":
                                print(f"      -> {result.get('result', '')[:60]}...")
                        else:
                            print(f"      -> Error: {result.get('error')}")

                # Add to conversation history.
                # Why assistant+user pair: mirrors the chat protocol.
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

        # Max iterations reached
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "partial_answer": response.final_answer if response else None,
            "tool_results": self.tool_results,
            "iterations": self.max_iterations
        }


# Demo Tasks
DEMO_TASKS = [
    {
        "name": "Gene Function",
        "task": "What does the BRCA1 protein do?",
        "expected_tools": ["search_uniprot", "get_protein_function"]
    },
    {
        "name": "Literature + Sequence",
        "task": "Find recent papers about TP53 and analyze this sequence: ATGGCTGACTACGTGGCTAGCCTGCCAGGCCGCTGCTGGCATGCTGACTACGTGGCTAGCTAA",
        "expected_tools": ["search_literature", "find_orfs", "translate_dna"]
    },
    {
        "name": "Comprehensive Gene Analysis",
        "task": "Tell me everything about the EGFR gene: what it does, its function, and any relevant papers.",
        "expected_tools": ["search_uniprot", "get_protein_function", "search_literature"]
    }
]


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 04: Real Agentic Workflow with API Tools")
    print("=" * 70)
    print("\nThis demo uses REAL bioinformatics APIs:")
    print("  - UniProt (https://uniprot.org) - protein database")
    print("  - PubMed (https://pubmed.ncbi.nlm.nih.gov) - literature")
    print("  - Local algorithms - sequence analysis")
    print()

    # Show available tasks
    print("Available demo tasks:")
    for i, task_info in enumerate(DEMO_TASKS, 1):
        print(f"  {i}. {task_info['name']}: {task_info['task']}")
    print()

    # Run first task
    # Why max_iterations=6: most tasks finish in 2-4 rounds. 6 gives headroom
    #   without letting the agent loop forever on a task it can't solve.
    task_info = DEMO_TASKS[0]
    print(f"Running: {task_info['name']}")
    print(f"Task: {task_info['task']}")
    print()

    agent = AgenticWorkflow(max_iterations=6, max_retries=2)
    result = agent.run(task_info['task'], verbose=True)

    # Summary
    print()
    print("=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"Status: {'Success' if result['success'] else 'Incomplete'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools called: {len(result['tool_results'])}")

    if result['tool_results']:
        print("\nTool execution trace:")
        for i, tr in enumerate(result['tool_results'], 1):
            status = "OK" if tr['result'].get('success') else "FAIL"
            print(f"  {i}. {status}: {tr['tool']}")

    print()
    print("=" * 70)
    print("What you saw:")
    print("  - Agent reflection before acting")
    print("  - REAL UniProt and PubMed API calls")
    print("  - Multi-step reasoning across iterations")
    print("  - Error handling and retry logic")
    print()
    print("This is production-grade agent architecture.")
    print("=" * 70)
