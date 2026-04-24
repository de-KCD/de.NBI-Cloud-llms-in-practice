---
title: "de.KCD / de.NBI Cloud LLMs in Practice"
subtitle: "From Prompts to Simple Agents in Bioinformatics"
version: "0.1"
date: "2026-04-20"
level: "Beginner to Advanced"
prerequisites:
  - "Python 3.9+"
  - "Basic command line and programming knowledge"
  - "LLM API access (provided)"
  - "Your own use cases!"
---

# Bioinformatics LLM Agent Demos - Hands-On Tutorial

## Overview
**Time estimation:** 3H+
**Last update:** 2026-04-20

**Questions:**
- What can LLMs do for bioinformatics, and where do they fail?
- How can I get structured, validated outputs from LLMs instead of free text?
- How do I build an autonomous agent that uses tools to solve tasks?
- How can I integrate real bioinformatics APIs with LLM reasoning?

**Objectives:**
- Explain the capabilities and limitations of LLMs for bioinformatics tasks.
- Use Pydantic models and instructor to obtain structured, validated JSON outputs.
- Build a tool-calling agent loop with system prompts and tool registration.
- Integrate external APIs (NCBI, UniProt) with autonomous LLM agents.
- Design and validate custom schemas for your own research data.
- Compare different agent architectures and choose appropriate patterns.

## Disclaimer:

Consistent with this tutorial's subject matter, LLMs were used to assist in preparing these materials. The human author retains responsibility for the underlying ideas, code, and experimental design. This work represents an experiment to evaluate how effectively open models can contribute to these tasks.

## Welcome

This tutorial walks you through actual working code in the `demo/` directory. We begin with simple LLM calls, then gradually introduce more agentic behaviors.

### Tutorial Structure

The demos are organized by chapter:

**Chapter 1: Basics (LLM Fundamentals)**
- demo_01: Basic LLM API calls
- demo_02: Structured outputs

**Chapter 2: Agents (Core Patterns)**
- demo_03: Simple tool-calling agent
- demo_04: Real bioinformatics APIs
- demo_05: Understanding LLM limitations
- demo_07: Full autonomous research agent

**Quick Navigation**

**Chapter 1: Basics**

| Demo | What It Does | Time | Complexity |
|------|--------------|------|------------|
| [demo_01](#demo-01-basic-llm-calls) | Basic LLM API calls | 10 min | Beginner |
| [demo_02](#demo-02-structured-outputs) | Structured JSON outputs | 20 min | Beginner |

**Chapter 2: Agents**

| Demo | What It Does | Time | Complexity |
|------|--------------|------|------------|
|[demo_03](#demo-03-simple-agent) | Simple tool-calling agent | 15 min | Beginner |
|[demo_04](#demo-04-real-api-agent) | Real bioinformatics APIs | 30 min | Intermediate |
|[demo_05](#demo-05-llm-limitations) | Understanding LLM limitations | 25 min | Intermediate |
|[demo_07](#demo-07-autonomous-research-agent) | Full autonomous research | 35 min | Advanced |

## Setup

### 1. Clone and Prepare

```bash
# Clone the tutorial files
git clone https://github.com/de-KCD/de.NBI-Cloud-llms-in-practice

# Navigate to the project directory
cd de.NBI-Cloud-llms-in-practice

# Check Python version (need 3.9+)
python --version

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install instructor requests openai pydantic
```

### 3. Configure API Access

```bash
# Set environment variables
export LLM_API_KEY="your-api-key-here"

# Are you using de.NBI Cloud VM to do this tutorial?
# Run ONLY ONE of them:
# YES:
export LLM_API_BASE="https://denbi-llm-api-internal.bihealth.org/v1"
# NO:
export LLM_API_BASE="https://denbi-llm-api.bihealth.org/v1"

export LLM_MODEL="qwen3.5-fp8"
```

> **Tip:** Add these to your `~/.bashrc` or `~/.zshrc` to persist across sessions.
> **Security:** Never commit API keys to version control. Use `.env` files or secret managers in production.

## Demo 01: Basic LLM Calls

**File:** `demo/demo_01_basic.py`

This demo covers making API calls to an LLM, exploring different types of tasks LLMs can perform, and understanding the limitations of unstructured outputs.

### Run the Demo

```bash
python demo/demo_01_basic.py
```

### What Happens

The demo makes four types of calls:

1. **Concept explanation** - "What is a reverse complement in DNA?"
2. **Calculation** - "What is the GC content of ATGCATGC?"
3. **Data transformation** - "Reverse complement this sequence..."
4. **Information extraction** - "Extract gene name, chromosome, exons from text"

### Key Concepts

**LLMs are text generators, not calculators:**

LLMs predict the next token based on training patterns. They do not actually compute or calculate -- they generate text that looks like calculations.

| Task Type | Example | Why It Works (or Doesn't) |
|-----------|---------|---------------------------|
| Explanation | "What is GC content?" | High - LLMs have seen many explanations |
| Calculation | "Calculate GC%" | Medium - Pattern-matching, not real math |
| Transformation | "Reverse complement..." | Medium - Common patterns work, long sequences fail |
| Extraction | "Extract gene info..." | Medium - Format varies, may miss fields |

Key insight: LLMs excel at tasks they have seen many times in training (explaining concepts, short sequence transformations). They struggle with novel calculations or long sequences where they cannot rely on patterns.

### Try It Yourself

Modify the demo to ask your own questions:

```python
# Add your own test
result = call_llm("What is the melting temperature of ATGCATGC?")
print(f"Answer: {result}")

# Or extract different info
text = "Sample SEQ001 has 45% GC content"
result = call_llm(f"Extract the sample ID and GC% from: {text}")
print(f"Extracted: {result}")
```

Notice: The LLM might make mistakes on calculations or transformations. This is exactly why **demo_02** introduces structured outputs with validation.

## Demo 02: Structured Outputs

**File:** `demo/demo_02_structured.py`

This demo shows how structured outputs solve demo_01's reliability problems, using Pydantic models for validation. It includes 9 bioinformatics schemas and demonstrates exporting results to DataFrame, CSV, and JSON. It also covers Instructor's JSON mode for grammar-constrained decoding.

### Run the Demo

```bash
python demo/demo_02_structured.py
```

### What Happens

The demo extracts structured data from text using 9 different schemas:

1. **GeneInfo** - Gene annotation with validation (UniProt pattern, exon range)
2. **ExperimentResult** - Lab measurements (custom validator for positive values)
3. **VariantCall** - VCF-like variant data (chromosome format validation)
4. **DifferentialExpression** - RNA-seq results (p-value range, direction check)
5. **PathwayEnrichment** - GO/KEGG enrichment (GO ID pattern validation)
6. **SequenceFeature** - ORFs, promoters, motifs (end > start validation)
7. **SampleMetadata** - Clinical/experimental metadata
8. **ClusterInfo** - scRNA-seq clusters (nested model with marker genes)
9. **GenomicRegion** - BED-like coordinates (0-based, end > start)

**Demo Flow:**

1. Show schema guide (which schema to use when)
2. Test each schema with realistic examples
3. Collect results (DE, variants)
4. Export to DataFrame, CSV, JSON
5. Point to exercises for hands-on practice

**Export Demo:**

At the end, the demo creates actual files:
- `demo02_de_results.json` - Differential expression data
- `demo02_variants.csv` - Variant calls (CSV format)
- `demo02_variants.json` - Variant calls (JSON format)

### Key Concepts

**Why structured outputs?**

Without structure (demo_01 problem):

```
"The BRCA1 gene has 24 exons on chromosome 17"
Format varies every time, hard to parse
LLM might make mistakes, no validation
```

With structure (demo_02 solution):

```python
GeneInfo(
    gene_name="BRCA1",      # Validated: uppercase, 1-20 chars
    exon_count=24,          # Validated: 1-10000
    chromosome="17",        # Optional
    uniprot_id="P38398"     # Validated: regex pattern
)
```

Result: Consistent, validated, ready to use.

**How instructor works:**

1. You define a Pydantic model (schema)
2. Instructor converts to JSON Schema
3. LLM generates JSON matching schema
4. Instructor parses, validates, and retries if needed
5. You receive a validated Pydantic object

**Why JSON MODE matters:**

| Feature | Without JSON Mode | With JSON Mode |
|------|--------------|------|
| Output format | LLM might return markdown | Only valid JSON |
| Explanations | Might include explanations | Pure data only |
| Structure | Can be malformed | Grammar-constrained |
| Parsing | You must parse manually | Auto-parsed |

### Try It Yourself

**Challenge:** Compare structured vs unstructured outputs:

```python
# WITHOUT structured output (demo_01)
from demo_01_basic import call_llm
text = "BRCA1 on chromosome 17 has 24 exons"
unstructured = call_llm(f"Extract gene, chr, exons from: {text}")
print(f"Unstructured: {unstructured}")
# Output varies: "BRCA1/17/24" or "Gene: BRCA1, Chr: 17..."
# Format different every time!

# WITH structured output (demo_02)
from demo_02_structured import extract_gene_info
structured = extract_gene_info(text)
print(f"Structured: {structured.gene_name}, {structured.chromosome}, {structured.exon_count}")
# Always: BRCA1, 17, 24
# Consistent, validated, ready to use!
```

**See the schema guide:**

Run `python demo/demo_02_structured.py` to see the full schema selection guide showing when to use each schema.

### Hands-On Exercises

Pick exercises for your field:

- **Bioinformatics:** Exercises 1-4
- **Chemistry:** Exercise 5
- **Clinical:** Exercise 6
- **Ecology:** Exercise 7
- **Neuroscience:** Exercise 8
- **Proteomics:** Exercise 9
- **Structural Biology:** Exercise 10

**Exercise 1: Simple Schema (5 min)**

Create a minimal schema and extraction function:

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

# Setup (same as demo_02)
client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY),
    mode=instructor.Mode.JSON
)

# Your schema
class MyResult(BaseModel):
    sample_id: str = Field(description="Sample identifier")
    gc_content: float = Field(description="GC percentage", ge=0, le=100)
    passed_qc: bool = Field(description="Quality control status")

# Extraction function
def extract_my_result(text: str) -> MyResult:
    return client.chat.completions.create(
        model=MODEL,
        response_model=MyResult,
        messages=[
            {"role": "system", "content": "Extract the data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )

# Test it
text = "Sample SEQ001 was analyzed. GC content measured at 52.3%."
result = extract_my_result(text)
print(f"Sample: {result.sample_id}, GC: {result.gc_content}%, QC: {result.passed_qc}")
```

**Exercise 2: Add Validation (10 min)**

Add pattern validation to ensure sample IDs follow a format:

```python
sample_id: str = Field(
    description="Sample identifier",
    pattern=r"^SEQ\d+$"  # Must be SEQ followed by digits
)
```

Test with invalid input to see validation catch errors:

```python
text = "Sample ABC123 has 50% GC"  # Should fail validation
try:
    result = extract_my_result(text)
except Exception as e:
    print(f"Validation caught error: {e}")
```

**Exercise 3: Nested Models (15 min)**

Create a parent-child relationship for sequencing data:

```python
from typing import List

class SequencingRead(BaseModel):
    read_id: str
    length: int = Field(ge=50)
    quality_score: float = Field(ge=0, le=40)

class SequencingRun(BaseModel):
    run_id: str
    platform: str
    reads: List[SequencingRead]  # Nested!
    mean_quality: float
```

**Exercise 4: Your Research (20 min)**

Think about your work. What data do you extract from papers or lab notebooks?

Examples by field:
- **Bioinformatics:** Primer design (Tm, GC%, length), antibody validation, cell line info
- **Chemistry:** Reaction conditions, compound properties, spectroscopy data
- **Clinical:** Patient demographics, treatment regimens, outcomes
- **Ecology:** Species counts, environmental parameters, GPS coordinates
- **Neuroscience:** Imaging parameters, task conditions, subject info
- **Proteomics:** MS run settings, sample prep, instrument config
- **Structural Biology:** PDB metadata, resolution, refinement stats

Create a schema for your use case and test it with real text.

**Exercises 5-10: Field-Specific Examples**

See `demo/demo_02_extensions.py` for 6 additional schemas:

| Exercise | Field | Schema | Example Task |
|------|--------------|------|---|
| **5** | Chemistry | `ChemicalCompound` | Extract MW, logP, CAS from paper |
| **6** | Clinical | `PatientSample` | Structure patient metadata |
| **7** | Ecology | `SpeciesObservation` | Parse field survey notes |
| **8** | Neuroscience | `ImagingSession` | Annotate fMRI/EEG sessions |
| **9** | Proteomics | `MassSpecRun` | Track LC-MS/MS runs |
| **10** | Structural Bio | `ProteinStructure` | Extract PDB metadata |

Run the examples:
```bash
python demo/demo_02_extensions.py
```

Then adapt the schema for your specific needs.

**Next:** demo_03 shows how to build an agent that uses tools autonomously.

## Demo 03: Simple Agent

**File:** `demo/demo_03_simple_agent.py`

**What You'll Learn:**
- Agent architecture (system prompt + tool loop)
- How tools are defined and registered
- Agent decision-making

### Run the Demo

```bash
python demo/demo_03_simple_agent.py
```

### What Happens

The agent tackles bioinformatics tasks by:
1. Reading the task
2. Deciding which tools to call
3. Executing tools
4. Synthesizing results into an answer

**Example interaction:**
```
Task: "What is the GC content of ATGCATGCATGC, and what is its reverse complement?"

Iteration 1:
  Tool: count_bases({"sequence": "ATGCATGCATGC"})
  Result: {"gc_content": 50.0, ...}

Iteration 2:
  Tool: reverse_complement({"sequence": "ATGCATGCATGC"})
  Result: {"result": "GCATGCATGCAT", ...}

Iteration 3:
  Task complete!
  Final Answer: "The GC content is 50.0% and the reverse complement is GCATGCATGCAT"
```

### Key Concepts

**The Agent Loop:**
1. Ask LLM what to do
2. If done=True: return answer
3. Execute tool calls
4. Add results to conversation
5. Repeat

### Try It Yourself

The demo includes 4 tools including `find_motif` for finding DNA motifs.

Try modifying the test tasks to explore different motifs:

```python
# Test with TATA box (promoter element)
task = "Find TATA box motifs in TATAAAGGCCATTATAA"
# Expected: 2 occurrences at positions 0 and 11

# Test with overlapping motifs
task = "Find AA in AAAA"
# Expected: 3 occurrences at positions 0, 1, 2 (overlapping!)

# Test with no matches
task = "Find GCGC in ATATATAT"
# Expected: 0 occurrences
```

**Add your OWN tool:**

Want to extend the agent? Add a tool that calculates melting temperature!

Step 1: Add the function (after `find_motif` function, around line 128):

```python
def calculate_tm(sequence: str) -> dict:
    """
    Calculate melting temperature (Tm) using simple rule:
    Tm = 2°C × (A+T) + 4°C × (G+C)

    This is the "wallace rule" - good for short sequences (<20bp)
    """
    try:
        seq = sequence.upper()
        a = seq.count('A')
        t = seq.count('T')
        g = seq.count('G')
        c = seq.count('C')
        tm = 2 * (a + t) + 4 * (g + c)
        return {"success": True, "tm": tm, "length": len(seq)}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

Step 2: Register the tool (in TOOLS dictionary, around line 182):

```python
TOOLS = {
    "reverse_complement": reverse_complement,
    "count_bases": count_bases,
    "explain_concept": explain_concept,
    "find_motif": find_motif,
    "calculate_tm": calculate_tm,  # <- Add this!
}
```

Step 3: Update system prompt (in system prompt, around line 234):

```python
- calculate_tm(sequence: str) - Calculate melting temperature
```

Step 4: Test it:

```python
print("--- Test 4: Melting Temperature ---")
task4 = "What is the melting temperature of ATGCATGC?"
print(f"Task: {task4}")
result4 = agent.run(task4)
print(f"Final Answer: {result4}")
```

This sequential combination (search then fetch) shows how agents chain tools to accomplish multi-step tasks.

The same pattern works for any domain: Examples:

| Field | Tool Idea | Purpose |
|------|--------------|------|
| **Chemistry** | `calculate_molarity()` | Compute molarity from mass/volume |
| **Chemistry** | `balance_equation()` | Balance chemical equations |
| **Clinical** | `calculate_bmi()` | BMI from height/weight |
| **Clinical** | `stage_cancer()` | TNM staging from tumor data |
| **Ecology** | `shannon_diversity()` | Calculate diversity index |
| **Ecology** | `gps_to_utm()` | Convert coordinates |
| **Neuroscience** | `calculate_snr()` | Signal-to-noise ratio |
| **Proteomics** | `peptide_mass()` | Calculate peptide mass |
| **Structural Bio** | `rmsd()` | RMSD between structures |

Example: Chemistry agent

```python
def calculate_molarity(mass: float, mw: float, volume: float) -> dict:
    """Calculate molarity from mass (g), MW (g/mol), volume (L)."""
    try:
        molarity = (mass / mw) / volume
        return {"success": True, "molarity": molarity, "unit": "M"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Register the tool
TOOLS["calculate_molarity"] = calculate_molarity

# Update system prompt
- calculate_molarity(mass, mw, volume) - Calculate solution molarity

# Test it
task = "What is the molarity of 5g NaCl (MW 58.44) in 0.5L water?"
result = agent.run(task)
```

**Exercise:** Add a tool for YOUR research workflow!

### Troubleshooting

**Agent loops forever:**

Increase `max_iterations` or simplify the task:

```python
agent = SimpleAgent(max_iterations=3)  # Limit to 3 iterations
```

**LLM calls wrong tool:**

Check that your tool descriptions in the system prompt are clear. The LLM decides based on:
- Tool name (should be descriptive)
- System prompt descriptions
- Task wording

**Tool returns unexpected format:**

Ensure all tools return `dict` with `"success": True/False`. The agent expects this pattern.

**Agent doesn't use new tool:**

1. Did you add it to `TOOLS` dict?
2. Did you update the system prompt?
3. Try explicitly mentioning the tool in your task: "Use calculate_tm to find..."

You now know:
- Agent architecture (system prompt + tool loop)
- How tools are defined and registered
- Agent decision-making

Next: demo_04 shows how to connect to real bioinformatics APIs.
- demo_05 -> LLM-generated tools (no hardcoded logic)

## Demo 04: Real API Agent

**File:** `demo/demo_04_real_api_agent.py`

**What You'll Learn:**
- Connecting to REAL bioinformatics APIs
- Multi-step reasoning workflows
- Agent reflection before acting

### Run the Demo

```bash
python demo/demo_04_real_api_agent.py
```

### What Happens

This agent connects to **real databases**:

1. **UniProt** - Protein database
2. **PubMed** - Scientific literature
3. **Sequence analysis** - Local computation

**Example task:**
```
"What does the BRCA1 protein do?"

Agent workflow:
1. REFLECT: "I need to find protein function"
2. PLAN: "Call search_uniprot first"
3. EXECUTE: search_uniprot("BRCA1") -> gets UniProt ID
4. EXECUTE: get_protein_function("P38398") -> gets functions
5. SYNTHESIZE: "BRCA1 is involved in DNA repair..."
```

### Key Concepts

**Reflection** - The agent thinks before acting:
```python
AgentReflection(
    what_have_we_learned="Found BRCA1 UniProt ID",
    what_still_unknown="Protein function unknown",
    next_step_rationale="Call get_protein_function next",
    confidence_in_answer=0.5,
    ready_to_conclude=False
)
```

### Available Tools

| Tool | What It Does | API |
|------|--------------|------|
| `search_uniprot` | Find protein by gene name | UniProt |
| `get_protein_function` | Get function annotations | UniProt |
| `search_literature` | Search PubMed papers | NCBI |
| `translate_dna` | DNA -> protein | Local |
| `find_orfs` | Find coding regions | Local |
| `reverse_complement` | Get reverse complement | Local |
| `count_bases` | Calculate GC content | Local |

### Try It Yourself

**Test different research questions:**

```python
agent = AgenticWorkflow(max_iterations=8)

# Protein function
result = agent.run("What is the function of TP53?")

# Literature search
result = agent.run("Find recent papers on CRISPR cancer therapy")

# Sequence analysis
result = agent.run("Translate ATGGCTGACTACGTA and find ORFs")

# Combine both
result = agent.run("What does EGFR do and what drugs target it?")
```

**Adjust for complex queries:**

For multi-part questions, increase iterations:

```python
# Complex question needs more steps
agent = AgenticWorkflow(max_iterations=12)
result = agent.run("Compare BRCA1 and TP53: functions, diseases, and recent papers")
```

**Add your own quick tool:**

Want to get the actual protein sequence from UniProt? The search gives metadata, but not the sequence itself!

```python
# Step 1: Add a function to extract sequence from UniProt
def get_protein_sequence(uniprot_id: str) -> dict:
    """Fetch the amino acid sequence for a UniProt ID."""
    try:
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}",
            params={"format": "json"},
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract sequence from nested structure
        sequence = data.get("sequence", {}).get("value", "")
        return {"success": True, "uniprot_id": uniprot_id, "sequence": sequence, "length": len(sequence)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Step 2: Add to TOOLS dict
TOOLS["get_protein_sequence"] = lambda uid: get_protein_sequence(uid)

# Step 3: Update TOOL_DESCRIPTIONS
TOOL_DESCRIPTIONS += "- get_protein_sequence(uniprot_id: str) - Get amino acid sequence\n"

# Step 4: Test it (combine with search_uniprot)
result = agent.run("Get the UniProt ID for BRCA1, then fetch its protein sequence")
```

**Why this matters:**
- **Sequence analysis** - Need actual sequence for BLAST, alignment, structure prediction
- **Complements search_uniprot** - search gives metadata, this gives the data
- **Simple extraction** - Just navigating UniProt's JSON structure
- **Real workflow** - Researchers always need sequences for downstream analysis

### Troubleshooting

**API connection fails:**

Check your internet and API endpoints:

```bash
# Test UniProt
curl "https://rest.uniprot.org/uniprotkb/search?query=gene:BRCA1&format=json&size=1"

# Test PubMed
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=BRCA1&retmode=json"
```

**Rate limiting:**

UniProt and PubMed have rate limits. If you hit them:
- Add delays between requests (`time.sleep(2)`)
- Reduce `max_results` parameter
- Use cached results for repeated queries

**Agent doesn't finish:**

Increase `max_iterations` for complex tasks:

```python
# Default is 8, try 12-15 for complex queries
agent = AgenticWorkflow(max_iterations=15)
```

**Tool returns error:**

Check the error message - common issues:
- Gene name not found -> Try alternative names (e.g., "TP53" vs "p53")
- Invalid sequence -> Ensure only A, T, G, C characters
- API downtime -> Retry later or use fallback tools

**LLM returns invalid JSON:**

This is rare with `instructor` but can happen. The demo uses `mode=instructor.Mode.JSON` which forces valid JSON. If you see parsing errors:
- Check `MAX_TOKENS` is high enough (2048+)
- Simplify the task
- Reduce `max_iterations`

### Comparison: Demo 03 vs 04 vs 05

For production decisions, consider these factors:

| Aspect | Demo 03 (Simple Agent) | Demo 04 (Real APIs) | Demo 05 (LLM-Generated) |
|------|--------------|------|---|
| **Tool Implementation** | Handwritten Python functions | API clients + local utils | LLM writes code at runtime |
| **Latency per Tool Call** | ~10ms (local) | 500-2000ms (network) | 3000-8000ms (LLM + exec) |
| **Accuracy** | 100% deterministic | API-dependent (95-99%) | ~85-95% (may hallucinate) |
| **Error Handling** | Try/except | HTTP retries, fallbacks | Syntax errors, validation needed |
| **Rate Limiting** | None | UniProt: generous, PubMed: 3/sec | LLM API limits apply |
| **Cost per Query** | $0.001-0.01 (LLM only) | $0.001-0.01 + free APIs | $0.01-0.05 (extra LLM calls) |
| **Debugging** | Stack traces, breakpoints | HTTP logs, API responses | Generated code + LLM behavior |
| **Testing** | Unit tests, mocks | Integration tests, API mocks | Hard to test (non-deterministic) |
| **Add New Tool** | 10-30 min (code + test) | 30-60 min (API research + code) | 5 min (description only) |
| **Maintenance** | Low (stable code) | Medium (API changes) | High (LLM updates, drift) |
| **Best Use Case** | Teaching, simple workflows | Production pipelines, real data | Prototyping, exploration |
| **Not Recommended For** | Production (too limited) | Time-critical applications | Validated/compliant workflows |
| **Validation Strategy** | Assert statements | Schema validation (Pydantic) | Post-execution verification |
| **Scalability** | High (local, parallel) | Medium (API rate limits) | Low (LLM bottleneck) |

**When to use which:**

- **Demo 03**: Learning agent patterns, testing logic, offline development
- **Demo 04**: Real research, production pipelines, reproducible results
- **Demo 05**: Rapid prototyping, exploring new tool ideas, one-off analysis

**Hybrid approach (recommended for production):**

Use Demo 04 architecture with:
- Core tools hardcoded (search_uniprot, translate_dna)
- LLM-generated tools for exploration (demo_05 pattern)
- Validation layer for LLM-generated outputs
- Caching for expensive API calls

You now know:
- How to connect to real bioinformatics APIs (UniProt, PubMed)
- Agent reflection pattern (think before acting)
- Multi-step reasoning workflows
- Error handling with retry logic
- Conversation memory across iterations
- When to use real APIs vs local computation

**Next:**
- demo_05 -> Understanding LLM limitations (why hybrid matters)
- demo_07 -> Full autonomous research agent

## Demo 05: Understanding LLM Limitations

> **WARNING:️  SKIP THIS DEMO IF:**
> - You already know LLMs don't actually calculate
> - You just want production patterns
> - You're short on time
>
> **READ THIS DEMO IF:**
> - You've wondered "why can't I just let the LLM compute?"
> - You want to understand the hybrid architecture rationale
> - You're building agents for real research (and need to know what NOT to do)

**File:** `demo/demo_05_llm_limitations.py`

**What You'll Learn:**
- Why LLMs are pattern-matchers, NOT calculators
- The danger of trusting LLM "confidence scores"
- When LLM pattern-matching works (and when it fails)
- Why hybrid architecture (demo_07) is the production solution

### Run the Demo

```bash
python demo/demo_05_llm_limitations.py
```

### WARNING:️  WARNING: Educational Only - Not for Production!

This demo shows what LLMs *appear* to do, but **DO NOT use this pattern for real research**:

- (not applicable) LLMs don't calculate - they pattern-match from training data
- (not applicable) "Confidence scores" are hallucinated (just another generated token)
- (not applicable) Arithmetic errors are common, especially with longer sequences
- (not applicable) Explanations are plausible reasoning, not actual computational steps

**This demo exists to teach you what NOT to do.**
```python
def count_bases(seq):
    # Human wrote this logic (demo_04 pattern)
    counts = {b: seq.count(b) for b in 'ATGC'}
    return counts
```

**Demo 05 (LLM pattern-matching - DON'T DO THIS):**
```python
# Human describes WHAT
LLM_TOOLS = {
    "count_bases": {
        "description": "Count each base and calculate GC content",
        "input": {"sequence": "DNA string"},
        "output": "Dict with counts and GC%"
    }
}

# LLM figures out HOW at runtime
# No code generation - LLM directly computes the answer!
```

### How It Actually Works

**Common misconception:** "LLM generates code -> exec()"

**Reality (safer!):** LLM directly computes with explanations:

```python
# 1. Agent decides to call count_bases
tool_call = {"tool_name": "count_bases", "arguments": {"sequence": "ATGC"}}

# 2. Look up tool description
tool_desc = LLM_TOOLS["count_bases"]

# 3. Send to LLM with computation prompt
response = client.chat.completions.create(
    model=MODEL,
    response_model=ToolResult,  # Structured output
    messages=[
        {"role": "system", "content": "You are a computational assistant..."},
        {"role": "user", "content": "Task: Count bases. Input: ATGC"}
    ]
)

# 4. LLM returns result WITH EXPLANATION
ToolResult(
    success=True,
    result={"A": 1, "T": 1, "G": 1, "C": 1, "gc_content": 50.0},
    explanation="Counted A=1, T=1, G=1, C=1. GC% = (1+1)/4*100 = 50%",
    confidence=0.95
)
```

**Why this is better than exec():**
- **Safe** - No arbitrary code execution
- **Explainable** - LLM generates plausible reasoning
- **Debuggable** - You can spot-check explanations
- **Flexible** - LLM adapts to edge cases
- WARNING:️  **NOT computation** - LLMs pattern-match, they don't calculate!

WARNING:️  **CRITICAL WARNING: LLMs Don't Actually Calculate!**

LLMs are **language models**, not calculators:
- They generate text that LOOKS LIKE calculations based on training patterns
- "Explanations" are plausible reasoning, not actual computational steps
- For simple patterns (GC% of short sequences), they've seen many examples
- For complex/long inputs, they're guessing based on patterns
- Confidence scores are hallucinated (just another generated token)
- Arithmetic errors are common, especially with longer sequences
- NEVER use for clinical, diagnostic, or production work
- ALWAYS verify with hardcoded tools (demo_04) for real research
- Use ONLY for: prototyping, education, exploration

### Available Tools

Demo 05 comes with **8 pre-built LLM tools**:

| Tool | Description |
|------|--------------|
| `reverse_complement` | Get reverse complement of DNA |
| `count_bases` | Count A,T,G,C + calculate GC% |
| `translate_dna` | DNA -> protein (genetic code) |
| `find_orfs` | Find open reading frames |
| `calculate_melting_temp` | Tm using Wallace rule |
| `gc_skew` | (G-C)/(G+C) ratio |
| `reverse_sequence` | Reverse only (no complement) |
| `complement_sequence` | Complement only (no reverse) |

### Try It Yourself

**Test different tools:**

```python
agent = LLMGeneratedToolAgent(max_iterations=8)

# Basic analysis
result = agent.run("Count bases in ATGCATGCATGC")

# Translation
result = agent.run("Translate ATGGCTGACTAC to protein")

# Advanced
result = agent.run("Calculate GC skew for GGGGCCCCAAAA")

# Multi-step
result = agent.run("Analyze ATGCATGC: GC%, reverse complement, Tm")
```

**Add your own tool (no coding!):**

Just add a description to `LLM_TOOLS`:

```python
LLM_TOOLS["motif_count"] = {
    "description": "Count occurrences of a DNA motif",
    "input": {
        "sequence": "DNA sequence",
        "motif": "Pattern to find (e.g., 'ATG')"
    },
    "output": "Count of motif occurrences"
}

# That's it! The LLM figures out the implementation.
# Test it:
result = agent.run("Count ATG motifs in ATGCATGCATGC")
```

**Compare with hardcoded tools:**

Run the same task on both demos:

```python
# Demo 04 (hardcoded)
# -> Fast, 100% accurate, no explanation

# Demo 05 (LLM-generated)
# -> Slower, ~95% accurate, full explanation
```

### Troubleshooting

WARNING:️  **FUNDAMENTAL LIMITATION: LLMs Don't Calculate, They Pattern-Match**

**What's actually happening:**

```python
# (not applicable)  WRONG mental model:
# LLM "counts" bases -> "calculates" GC% -> "shows work"

#  CORRECT mental model:
# LLM generates text that looks like:
#   "Counted G=2, C=2, GC% = 50%"
# based on patterns it's seen in training
```

**Confidence scores are meaningless:**

```python
# WARNING:️  This is DANGEROUS - confidence is hallucinated!
if result.confidence < 0.8:  # DON'T DO THIS
    print("Low confidence")  # The 0.95 is just as made up!
```

The "confidence" field is **another LLM generation**, not real statistical confidence.
An LLM can confidently say 0.95 while being completely wrong.

**Explanations aren't actual work:**

```python
# (not applicable)  WRONG: "The explanation shows the calculation steps"
#  CORRECT: "The explanation is plausible-sounding reasoning"

# The LLM didn't actually count anything.
# It generated text that looks like counting.
```

**What to do instead:**

```python
#  CORRECT: Verify with hardcoded tools
llm_result = agent.run("Calculate GC content of ATGCATGC")
hardcoded_result = count_bases("ATGCATGC")  # demo_04 tool

if llm_result != hardcoded_result:
    print("WARNING:️  LLM pattern-matched incorrectly!")
```

**When LLMs do well:**
- Short sequences (<50 bases)
- Common patterns (GC%, reverse complement)
- Tasks seen frequently in training data

**When LLMs fail:**
- Long sequences (>100 bases) - they lose track
- Unusual calculations - no training patterns
- Multi-step arithmetic - compounding errors
- Edge cases - rare patterns

**Solution:** Use demo_04 (hardcoded) for ALL real calculations.
Demo_036 is for **prototyping and education only**.

**LLM doesn't understand tool description:**

Be more explicit in `LLM_TOOLS`:

```python
# Vague (bad)
"description": "Analyze sequence"

# Specific (good)
"description": "Count A,T,G,C bases and calculate GC percentage"
```

**Multi-step tasks fail:**

Increase iterations and break down the task:

```python
# Too complex for few iterations
agent = LLMGeneratedToolAgent(max_iterations=12)
```

### When to Use LLM-Generated vs Hardcoded

| Scenario | Use Hardcoded (04) | Use LLM-Generated (05) |
|------|--------------|------|
| Clinical diagnostics | Yes | (not applicable) **DANGEROUS** |
| Production pipeline | Yes | No |
| High-throughput (1000s of sequences) | Yes | No |
| Scientific publication | Yes | (not applicable) **Must verify** |
| Rapid prototyping | No | Yes |
| Educational demos | No | **Show limitations!** |
| One-off exploration | No | Yes |
| Need explanations | No | **But verify!** |
| Understanding LLM capabilities | No | **Educational** |

WARNING:️  **Golden rule:** If it matters scientifically, verify with hardcoded tools!

**Why LLMs aren't calculators:**

LLMs are **language models** trained to predict text, not perform calculations:
- They've seen "GC% of ATGC is 50%" many times in training
- For short, common patterns, they generate correct-looking answers
- For long sequences or novel calculations, they're guessing based on patterns
- The "explanation" is plausible reasoning, not actual computational steps
- This is fine for prototyping and education, but NOT for real science

You now know:
- LLM as pattern-matcher (not computation)
- Tools as descriptions, not functions
- Direct text generation (safer than exec!)
- Explanations are plausible reasoning, not actual work
WARNING:️  **Confidence scores are hallucinated - NEVER trust them!**
WARNING:️  **LLMs don't calculate - they pattern-match from training!**
WARNING:️  **Verify everything with hardcoded tools for real science!**
- Trade-offs: flexibility vs accuracy vs honesty
- When to use hardcoded vs LLM-generated (and when NOT to)

**Critical takeaways:**
1. LLMs generate text that LOOKS LIKE calculations
2. Explanations are post-hoc rationalizations, not actual work
3. Demo_05 is for **prototyping and education ONLY**
4. For real science: **demo_04 (hardcoded) or verify independently**
5. Be honest about what LLMs can and cannot do

**Next:**
- demo_07 -> Full autonomous research agent

## Demo 07: Full Autonomous Research Agent

**File:** `demo/demo_07_autonomous_research_agent.py`

The final evolution: an agent that combines **real APIs**, **deterministic algorithms**, and **LLM reasoning** into a fully autonomous research system.

### What Makes This Different

| Demo | Tools | Autonomy | Production-Ready |
|------|-------|----------|------------------|
| demo_03 | 4 local tools | Limited | No |
| demo_04 | APIs only | Fixed workflow | Maybe |
| demo_05 | LLM-generated | N/A | Never |
| **demo_07** | **APIs + Algorithms + Knowledge** | **Full** | **Yes** |

### Tool Catalog

**Algorithms (5)** - 100% deterministic Python:
- `count_bases`, `find_orfs`, `search_motif`, `find_cpg_islands`, `reverse_complement`

**APIs (6)** - Real HTTP calls:
- `search_uniprot`, `get_protein_function`, `search_pubmed`, `fetch_gene_sequence`, `get_protein_sequence`

**Knowledge (2)** - LLM reasoning:
- `interpret_motif`, `evaluate_coding_potential`

### Architecture

```
+-----------+     +----------------+     +-----------+
|   LLM     | --> |   Decision:    | --> |  Tool     |
|  (brain)  |     | Call tool or   |     | Execution |
+-----------+     | Conclude       |     +-----------+
      ^            +----------------+          |
      |                    |                   |
      |                    v                   |
      |             +----------------+         |
      +-------------| Results feed   | <-------+
                    | back to LLM    |
                    +----------------+
```

### Run the Demo

```bash
python demo/demo_07_autonomous_research_agent.py
```

### Example Tasks

**Task 1: Fetch and analyze TERT promoter**
```
Agent decides:
1. fetch_gene_sequence(TERT, promoter) -> gets 2500bp
2. count_bases + find_orfs + search_motif(TATA box)
3. search_pubmed(TERT promoter cancer) -> gets citations
4. interpret_motif(TATA box position) -> biological significance
5. Conclude: "TERT promoter contains core promoter elements, 
    associated with cancer via telomerase reactivation"
```

**Task 2: BRCA1 protein function**
```
Agent decides:
1. search_uniprot(BRCA1) -> P38398
2. get_protein_function(P38398) -> DNA repair
3. search_pubmed(BRCA1 breast cancer) -> 5 papers
4. Conclude with citations
```

### Key Features

1. **No hardcoded workflows** - LLM decides what to do
2. **Real citations** - PubMed papers with PMIDs
3. **Hypothesis tracking** - Agent maintains working hypothesis
4. **Alternative explanations** - Scientific skepticism built-in
5. **Error handling** - Graceful failures, partial results

### When to Use

| Use Case | Good Fit? |
|----------|-----------|
| Open-ended research questions | Yes |
| Literature + sequence analysis | Yes |
| Exploratory hypothesis generation | Yes |
| Batch processing 1000s of sequences | No (use scripted) |
| Clinical diagnostics | No (verify everything) |

### Troubleshooting

**Agent loops too long:**
- Reduce `max_iterations` (default: 10)
- Make task more specific

**API calls fail:**
- Check internet connection
- Verify UniProt/NCBI availability
- Look at `success: False` in results

**LLM ignores knowledge tools:**
- Tools are optional - LLM may not need them
- Explicitly ask: "Interpret the biological significance"

**Output too verbose:**
- Set `verbose=False` in investigate()
- Review JSON report instead of printed output

You now know the complete progression:
- demo_01-02: LLM fundamentals
- demo_03: Simple agents
- demo_04: External APIs
- demo_05: What NOT to do
- demo_07: Full autonomy (APIs + algorithms + reasoning)

## Learning Path

### For Beginners

Start here:
1. demo_01 -> Understand LLM basics
2. demo_02 -> Structured outputs
3. demo_03 -> See simple agent

### For Intermediate Users

Try these:
1. demo_04 -> Real APIs
2. demo_05 -> LLM-generated tools
3. demo_07 -> Full autonomous research agent
4. Modify and extend!

### For Advanced Users

Explore:
1. demo_07 -> Full autonomy
2. Modify and extend!

## Troubleshooting

### API Connection Issues

```bash
# Test API connection
curl -H "Authorization: Bearer $LLM_API_KEY" \
     https://denbi-llm-api.bihealth.org/v1/models
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade instructor requests openai pydantic
```

### Agent Stuck in Loop

- Increase `max_iterations`
- Simplify the task
- Check tool definitions

## Next Steps

You now know:
- Basic LLM API calls (demo_01)
- Structured outputs with validation (demo_02)
- Simple agent architecture (demo_03)
- Real bioinformatics APIs (demo_04)
- LLM pattern-matching limitations (demo_05)
- Full autonomous research with real APIs (demo_07)

### Where to Go Next

1. **Build your own tools** - Add domain-specific functions
2. **Combine demos** - Use memory + batch + real APIs
3. **Adapt to your workflow** - Modify for your research
4. **Share back** - Contribute improvements!

### Resources

- [Instructor docs](https://python.useinstructor.com/)
- [Pydantic docs](https://docs.pydantic.dev/)
- [UniProt API](https://www.uniprot.org/help/api_queries)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)

## Get Help

- Check existing issues in the repository
- Read the code comments (they're detailed!)
- Experiment and learn by doing
