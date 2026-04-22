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

---

## Overview
**Time estimation:** 3-4H
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

---

## Disclaimer:

In keeping with this tutorial's subject matter, LLMs were used to format and structure this document. The human author retains full responbility of the underlying ideas, code, and experimental design. The AI's role was limited to polishing messy drafts into a clear, readable narrative for participants.

## Welcome! 👋

This tutorial walks you through **actual working code** in the `demo/` directory. Unlike theoretical tutorials, you'll run real agents that analyze DNA sequences, search scientific databases, and generate insights autonomously.

### What You'll Find Here

The demos are organized by **chapter** and **theme**:

```
Chapter 1: Basics (LLM Fundamentals)
┌─────────────────────────────────────────┐
│  demo_01 → demo_02                      │
│  (basic calls) → (structured outputs)   │
└─────────────────────────────────────────┘

Chapter 2: Agents (Core Patterns)
┌─────────────────────────────────────────────────────────────────┐
│  demo_03 → demo_04 → demo_05 → demo_06                          │
│  (simple    (real     (LLM      (hybrid                         │
│   agent)   APIs)     generated) autonomous)                     │
└─────────────────────────────────────────────────────────────────┘

Chapter 3: Tool Development (Appendix - Optional Deep Dives)
┌──────────────────────────────────────────────────────────────────────┐
│  demo_07 (includes detective mode)                                   │
│  (sequence characterization + hypothesis-driven investigation)       │
└──────────────────────────────────────────────────────────────────────┘
```

### Quick Navigation

**Chapter 1: Basics**

| Demo | What It Does | Time | Complexity |
|------|--------------|------|------------|
| [demo_01](#demo-01-basic-llm-calls) | Basic LLM API calls | 10 min | ⭐ |
| [demo_02](#demo-02-structured-outputs) | Structured JSON outputs | 20 min | ⭐⭐ |

**Chapter 2: Agents**

| Demo | What It Does | Time | Complexity |
|------|--------------|------|------------|
| [demo_03](#demo-03-simple-agent) | Simple tool-calling agent | 15 min | ⭐⭐ |
| [demo_04](#demo-04-real-api-agent) | Real bioinformatics APIs | 30 min | ⭐⭐⭐ |
| [demo_05](#demo-05-llm-limitations) | Understanding LLM limitations | 25 min | ⭐⭐⭐ |
| [demo_06](#demo-06-hybrid-autonomous) | Hybrid autonomous agent | 30 min | ⭐⭐⭐⭐ |

**Chapter 3: Tool Development (Optional)**

| Demo | What It Does | Time | Complexity |
|------|--------------|------|------------|
| [demo_07](#demo-07-sequence-characterization) | Sequence analysis pipeline | 20 min | ⭐⭐⭐ |

> **Includes:** Detective mode (`demo_07b_detective.py`) - hypothesis-driven investigation with NCBI BLAST

---

## Setup

### 1. Clone and Prepare

```bash
# Clone the tutorial files
git clone

# Navigate to the project directory
cd /path/to/your/project

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
>
> **Security:** Never commit API keys to version control. Use `.env` files or secret managers in production.

---

## Demo 01: Basic LLM Calls

**File:** `demo/demo_01_basic.py`

**What You'll Learn:**
- How to make API calls to an LLM
- Different types of tasks LLMs can perform
- Limitations of unstructured outputs

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

LLMs predict the next token based on training patterns. They don't actually "compute" or "calculate" - they generate text that looks like calculations.

| Task Type | Example | Why It Works (or Doesn't) |
|-----------|---------|---------------------------|
| Explanation | "What is GC content?" | ✓ High - LLMs have seen many explanations |
| Calculation | "Calculate GC%" | ⚠ Medium - Pattern-matching, not real math |
| Transformation | "Reverse complement..." | ⚠ Medium - Common patterns work, long sequences fail |
| Extraction | "Extract gene info..." | ⚠ Medium - Format varies, may miss fields |

> **Key insight:** LLMs excel at tasks they've seen many times in training (explaining concepts, short sequence transformations). They struggle with novel calculations or long sequences where they can't rely on patterns.

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

> **⚠ Notice:** The LLM might make mistakes on calculations or transformations!
> This is exactly why **demo_02** introduces structured outputs with validation.

---

## Demo 02: Structured Outputs

**File:** `demo/demo_02_structured.py`

**What You'll Learn:**
- How structured outputs solve demo_01's reliability problems
- Using Pydantic models for validation
- 9 bioinformatics schemas
- Export to DataFrame, CSV, JSON
- Instructor's JSON mode (grammar-constrained decoding)

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
8. **ClusterInfo** - scRNA-seq clusters (nested model with marker genes!)
9. **GenomicRegion** - BED-like coordinates (0-based, end > start)

**Demo Flow:**

```
┌─────────────────────────────────────────────────────────┐
│  1. Show schema guide (which schema to use when)        │
│  2. Test each schema with realistic examples            │
│  3. Collect results (DE, variants)                      │
│  4. Export to DataFrame, CSV, JSON                      │
│  5. Point to exercises for hands-on practice            │
└─────────────────────────────────────────────────────────┘
```

**Export Demo:**

At the end, the demo creates actual files:
- `demo02_de_results.json` - Differential expression data
- `demo02_variants.csv` - Variant calls (CSV format)
- `demo02_variants.json` - Variant calls (JSON format)

### Key Concepts

**Why structured outputs?**

❌ Without structure (demo_01 problem):
```
"The BRCA1 gene has 24 exons on chromosome 17"
→ Format varies every time, hard to parse
→ LLM might make mistakes, no validation
```

✅ With structure (demo_02 solution):
```python
GeneInfo(
    gene_name="BRCA1",      # Validated: uppercase, 1-20 chars
    exon_count=24,          # Validated: 1-10000
    chromosome="17",        # Optional
    uniprot_id="P38398"     # Validated: regex pattern
)
→ Consistent, validated, ready to use!
```

**How instructor works:**

```
┌─────────────────────────────────────────────────────────┐
│  1. You define Pydantic model (schema)                  │
│         ↓                                               │
│  2. instructor converts to JSON Schema                  │
│         ↓                                               │
│  3. LLM generates JSON matching schema                  │
│         ↓                                               │
│  4. instructor parses → validates → retries if needed   │
│         ↓                                               │
│  5. You get validated Pydantic object                   │
└─────────────────────────────────────────────────────────┘
```

**Why JSON MODE matters:**

| Without JSON Mode | With JSON Mode |
|-------------------|----------------|
| LLM might return markdown | Only valid JSON |
| Might include explanations | Pure data only |
| Can be malformed | Grammar-constrained |
| You must parse manually | Auto-parsed |

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

**Pick exercises for YOUR field!**

> **Bioinformatics:** Exercises 1-4
> **Chemistry:** Exercise 5
> **Clinical:** Exercise 6
> **Ecology:** Exercise 7
> **Neuroscience:** Exercise 8
> **Proteomics:** Exercise 9
> **Structural Biology:** Exercise 10

**Exercise 1: Simple Schema (5 min) - ALL FIELDS**

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
    print(f"✓ Validation caught error: {e}")
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

**Exercise 4: Your Research (20 min) - ALL FIELDS**

Think about YOUR work. What data do you extract from papers or lab notebooks?

Examples:
- **Bioinformatics**: Primer design (Tm, GC%, length), antibody validation, cell line info
- **Chemistry**: Reaction conditions, compound properties, spectroscopy data
- **Clinical**: Patient demographics, treatment regimens, outcomes
- **Ecology**: Species counts, environmental parameters, GPS coordinates
- **Neuroscience**: Imaging parameters, task conditions, subject info
- **Proteomics**: MS run settings, sample prep, instrument config
- **Structural Biology**: PDB metadata, resolution, refinement stats

Create a schema for your use case and test it with real text!

**Exercises 5-10: Field-Specific Examples**

See `demo/demo_02_extensions.py` for 6 additional schemas:

| Exercise | Field | Schema | Example Task |
|----------|-------|--------|--------------|
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

Then adapt the schema for YOUR specific needs!

**What You've Learned:**

✅ Why structured outputs matter (demo_01's problem → demo_02's solution)
✅ How instructor works (Pydantic → JSON Schema → validated object)
✅ 9 bioinformatics schemas + 6 interdisciplinary schemas
✅ Validation types: Field constraints, patterns, custom validators, nested models
✅ Export to DataFrame, CSV, JSON
✅ How to create your own schemas for YOUR field

**Next:** demo_03 shows how to build an **agent** that uses tools autonomously!

---

## Demo 03: Simple Agent

**File:** `demo/demo_03_agent.py`

**What You'll Learn:**
- Agent architecture (system prompt + tool loop)
- How tools are defined and registered
- Agent decision-making

### Run the Demo

```bash
python demo/demo_03_agent.py
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
  ✓ Task complete!
  Final Answer: "The GC content is 50.0% and the reverse complement is GCATGCATGCAT"
```

### Key Concepts

**The Agent Loop:**
```
┌─────────────────────────────────────────────┐
│  FOR each iteration:                        │
│    1. Ask LLM what to do                    │
│    2. If done=True: return answer           │
│    3. Execute tool calls                    │
│    4. Add results to conversation           │
│    5. Repeat                                │
└─────────────────────────────────────────────┘
```

### Try It Yourself

**The demo now comes with 4 tools** including `find_motif` for finding DNA motifs!

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

**Step 1: Add the function** (after `find_motif` function, around line 128):

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

**Step 2: Register the tool** (in TOOLS dictionary, around line 182):

```python
TOOLS = {
    "reverse_complement": reverse_complement,
    "count_bases": count_bases,
    "explain_concept": explain_concept,
    "find_motif": find_motif,
    "calculate_tm": calculate_tm,  # ← Add this!
}
```

**Step 3: Update system prompt** (in system prompt, around line 234):

```python
- calculate_tm(sequence: str) - Calculate melting temperature
```

**Step 4: Test it:**

```python
print("--- Test 4: Melting Temperature ---")
task4 = "What is the melting temperature of ATGCATGC?"
print(f"Task: {task4}")
result4 = agent.run(task4)
print(f"Final Answer: {result4}")
```

**Why this matters:**

Finding motifs and calculating Tm are essential for:
- **PCR primer design** - Need right Tm for annealing
- **Gene prediction** - Start/stop codons mark gene boundaries
- **Regulatory analysis** - Promoters contain specific motifs
- **Mutation detection** - Changes in motifs can disrupt function

**Build Tools for YOUR Field:**

The same pattern works for ANY domain! Examples:

| Field | Tool Idea | What It Does |
|-------|-----------|--------------|
| **Chemistry** | `calculate_molarity()` | Compute molarity from mass/volume |
| **Chemistry** | `balance_equation()` | Balance chemical equations |
| **Clinical** | `calculate_bmi()` | BMI from height/weight |
| **Clinical** | `stage_cancer()` | TNM staging from tumor data |
| **Ecology** | `shannon_diversity()` | Calculate diversity index |
| **Ecology** | `gps_to_utm()` | Convert coordinates |
| **Neuroscience** | `calculate_snr()` | Signal-to-noise ratio |
| **Proteomics** | `peptide_mass()` | Calculate peptide mass |
| **Structural Bio** | `rmsd()` | RMSD between structures |

**Example: Chemistry agent**

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

### What You've Learned

✅ Agent architecture (system prompt + tool loop)
✅ How tools extend agent capabilities
✅ Multi-step reasoning (agent chains tool calls)
✅ Tool design patterns (return dict with success status)
✅ How to add your own tools

**Next:**
- demo_04 → Real API tools (UniProt, PubMed, BLAST)
- demo_05 → LLM-generated tools (no hardcoded logic)

---

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
3. EXECUTE: search_uniprot("BRCA1") → gets UniProt ID
4. EXECUTE: get_protein_function("P38398") → gets functions
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
|------|--------------|-----|
| `search_uniprot` | Find protein by gene name | UniProt |
| `get_protein_function` | Get function annotations | UniProt |
| `search_literature` | Search PubMed papers | NCBI |
| `translate_dna` | DNA → protein | Local |
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
- Gene name not found → Try alternative names (e.g., "TP53" vs "p53")
- Invalid sequence → Ensure only A, T, G, C characters
- API downtime → Retry later or use fallback tools

**LLM returns invalid JSON:**

This is rare with `instructor` but can happen. The demo uses `mode=instructor.Mode.JSON` which forces valid JSON. If you see parsing errors:
- Check `MAX_TOKENS` is high enough (2048+)
- Simplify the task
- Reduce `max_iterations`

### Comparison: Demo 03 vs 04 vs 05

For production decisions, consider these factors:

| Aspect | Demo 03 (Simple Agent) | Demo 04 (Real APIs) | Demo 05 (LLM-Generated) |
|--------|------------------------|----------------------|--------------------------|
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

### What You've Learned

✅ How to connect to real bioinformatics APIs (UniProt, PubMed)
✅ Agent reflection pattern (think before acting)
✅ Multi-step reasoning workflows
✅ Error handling with retry logic
✅ Conversation memory across iterations
✅ When to use real APIs vs local computation

**Next:**
- demo_05 → Understanding LLM limitations (why hybrid matters)
demo_06 → Fully autonomous investigation

---

## Demo 05: Understanding LLM Limitations

> **⚠️  SKIP THIS DEMO IF:**
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
- Why hybrid architecture (demo_06) is the production solution

### Run the Demo

```bash
python demo/demo_05_llm_limitations.py
```

### ⚠️  WARNING: Educational Only - Not for Production!

This demo shows what LLMs *appear* to do, but **DO NOT use this pattern for real research**:

- ❌ LLMs don't calculate - they pattern-match from training data
- ❌ "Confidence scores" are hallucinated (just another generated token)
- ❌ Arithmetic errors are common, especially with longer sequences
- ❌ Explanations are plausible reasoning, not actual computational steps

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

**Common misconception:** "LLM generates code → exec()"

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
- ✅ **Safe** - No arbitrary code execution
- ✅ **Explainable** - LLM generates plausible reasoning
- ✅ **Debuggable** - You can spot-check explanations
- ✅ **Flexible** - LLM adapts to edge cases
- ⚠️  **NOT computation** - LLMs pattern-match, they don't calculate!

⚠️  **CRITICAL WARNING: LLMs Don't Actually Calculate!**

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
|------|-------------|
| `reverse_complement` | Get reverse complement of DNA |
| `count_bases` | Count A,T,G,C + calculate GC% |
| `translate_dna` | DNA → protein (genetic code) |
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
# → Fast, 100% accurate, no explanation

# Demo 05 (LLM-generated)
# → Slower, ~95% accurate, full explanation
```

### Troubleshooting

⚠️  **FUNDAMENTAL LIMITATION: LLMs Don't Calculate, They Pattern-Match**

**What's actually happening:**

```python
# ❌  WRONG mental model:
# LLM "counts" bases → "calculates" GC% → "shows work"

# ✅  CORRECT mental model:
# LLM generates text that looks like:
#   "Counted G=2, C=2, GC% = 50%"
# based on patterns it's seen in training
```

**Confidence scores are meaningless:**

```python
# ⚠️  This is DANGEROUS - confidence is hallucinated!
if result.confidence < 0.8:  # DON'T DO THIS
    print("Low confidence")  # The 0.95 is just as made up!
```

The "confidence" field is **another LLM generation**, not real statistical confidence.
An LLM can confidently say 0.95 while being completely wrong.

**Explanations aren't actual work:**

```python
# ❌  WRONG: "The explanation shows the calculation steps"
# ✅  CORRECT: "The explanation is plausible-sounding reasoning"

# The LLM didn't actually count anything.
# It generated text that looks like counting.
```

**What to do instead:**

```python
# ✅  CORRECT: Verify with hardcoded tools
llm_result = agent.run("Calculate GC content of ATGCATGC")
hardcoded_result = count_bases("ATGCATGC")  # demo_04 tool

if llm_result != hardcoded_result:
    print("⚠️  LLM pattern-matched incorrectly!")
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
|----------|---------------------|-------------------------|
| Clinical diagnostics | ✅ | ❌ **DANGEROUS** |
| Production pipeline | ✅ | ❌ |
| High-throughput (1000s of sequences) | ✅ | ❌ |
| Scientific publication | ✅ | ❌ **Must verify** |
| Rapid prototyping | ❌ | ✅ |
| Educational demos | ❌ | ✅ **Show limitations!** |
| One-off exploration | ❌ | ✅ |
| Need explanations | ❌ | ✅ **But verify!** |
| Understanding LLM capabilities | ❌ | ✅ **Educational** |

⚠️  **Golden rule:** If it matters scientifically, verify with hardcoded tools!

**Why LLMs aren't calculators:**

LLMs are **language models** trained to predict text, not perform calculations:
- They've seen "GC% of ATGC is 50%" many times in training
- For short, common patterns, they generate correct-looking answers
- For long sequences or novel calculations, they're guessing based on patterns
- The "explanation" is plausible reasoning, not actual computational steps
- This is fine for prototyping and education, but NOT for real science

### What You've Learned

✅ LLM as pattern-matcher (not computation)
✅ Tools as descriptions, not functions
✅ Direct text generation (safer than exec!)
✅ Explanations are plausible reasoning, not actual work
⚠️  **Confidence scores are hallucinated - NEVER trust them!**
⚠️  **LLMs don't calculate - they pattern-match from training!**
⚠️  **Verify everything with hardcoded tools for real science!**
✅ Trade-offs: flexibility vs accuracy vs honesty
✅ When to use hardcoded vs LLM-generated (and when NOT to)

**Critical takeaways:**
1. LLMs generate text that LOOKS LIKE calculations
2. Explanations are post-hoc rationalizations, not actual work
3. Demo_05 is for **prototyping and education ONLY**
4. For real science: **demo_04 (hardcoded) or verify independently**
5. Be honest about what LLMs can and cannot do

### 🏆 Agent Olympics - See It In Action!

Want to see the difference between demo_04, demo_05, and demo_06? Run the comparison script:

```bash
python scripts/agent_olympics.py
```

This runs the **same task** on all three architectures and compares:
- **Accuracy** - Does it get the right answer?
- **Speed** - How long does it take?
- **Reliability** - Does it fail on long sequences?

**Expected output:**

```
Test: Short (8bp) sequence
Expected GC%: 50.0%

demo_04 (Real API):     50.0%  ✓  0.8s
demo_05 (LLM pattern):  50.0%  ✓  3.2s  ⚠️  got lucky!
demo_06 (Hybrid):       50.0%  ✓  1.5s

Test: Long (200bp) sequence
Expected GC%: 50.0%

demo_04 (Real API):     50.0%  ✓
demo_05 (LLM pattern):  48.0%  ✗  (LLM lost track!)
demo_06 (Hybrid):       50.0%  ✓
```

**Key insight:** LLMs can handle short, common patterns. They fail on longer sequences where they can't rely on training data.

**Next:**
- demo_06 → Fully autonomous investigation

---

## Demo 07: Sequence Characterization (Appendix)

**File:** `demo/demo_07_sequence_characterization.py`

### Where This Fits in the Progression

```
demo_01 → Basic LLM calls (what can LLMs do?)
demo_02 → Structured outputs (reliable parsing)
demo_03 → Simple agent (tool orchestration)
demo_04 → Real API agent (external databases)
demo_05 → LLM-generated tools (flexible but unreliable)
demo_06 → Autonomous investigation (agent + analysis)
```

**Key insight:** Demo 07 is an **optional deep-dive** into sequence analysis pipelines. Unlike the agent demos (03-06), it skips the LLM entirely and provides a **real bioinformatics pipeline** you can use in actual research.

### What You'll Learn

- **6 analysis components** in one comprehensive pipeline
- **k-mer frequency analysis** (sequence signatures)
- **Restriction enzyme mapping** (for cloning)
- **ORF prediction** (finding genes)
- **Feature detection** (promoters, CpG islands)
- **Comparative analysis** (vs random sequences)
- **When to use pipelines vs agents**

### Why Skip the Agent?

This demo asks: **"Do we always need an LLM?"**

**Answer:** No! For routine sequence analysis:

| Aspect | Agent (03/04/05/06) | Pipeline (07) |
|--------|-------------------|---------------|
| Speed | 500ms-8s per sequence | <10ms per sequence |
| Cost | API calls ($0.001-0.05) | Free (local) |
| Accuracy | 85-100% (varies) | 100% deterministic |
| Privacy | Sequences sent to API | All local |
| Scalability | Slow for 1000s | Batch process easily |
| Flexibility | High (adapts to questions) | Fixed analysis |
| Best for | Exploration, Q&A | Production, batch |

**Rule of thumb:**
- **Exploration** ("What's interesting about this sequence?") → Use agents (04/05/06)
- **Production** ("Analyze 500 sequences for my paper") → Use pipeline (07)

### Run the Demo

```bash
python demo/demo_07_sequence_characterization.py
```

### What You Get

A **complete analysis report** for each sequence:

```
SEQUENCE: Promoter + start codon region
Input: TATAAAGGCCACCATGGCTGACTACGTAGCTAG

--- Quick Summary ---
Length: 33 bp
GC Content: 51.52%
ORFs found: 1
Restriction sites: 3 enzymes
TATA box: YES (at position 0)
k-mer correlation vs random: 0.85
Interpretation: Non-random sequence with regulatory features
```

### Analysis Components (Deep Dive)

#### 1. Basic Statistics

**What it computes:**
- Length (bp)
- GC content (%)
- Base composition (A%, T%, G%, C%)

**Biological meaning:**
- **GC%** affects DNA stability (higher GC = higher melting temp)
- **Base composition** biases reveal evolutionary pressures
- **Length** determines analysis approach (short reads vs long reads)

**Lab applications:**
- PCR primer design (need specific GC% and Tm)
- Sequencing quality control (unexpected composition = contamination)

#### 2. k-mer Analysis

**What it computes:**
- Frequency of all k-mers (k=1,2,3,4)
- Comparison to expected random frequencies
- Correlation coefficient

**Biological meaning:**
- **Non-random patterns** = functional elements
- **CpG suppression** = methylation signatures
- **Codon bias** = organism-specific translation efficiency

**Example:**
```
Sequence: ATGCATGC
1-mer: A=25%, T=25%, G=25%, C=25% (uniform)
2-mer: AT=25%, TG=25%, GC=25%, CA=25% (repeating pattern)

Correlation vs random: 0.95 → Strong non-random structure!
```

#### 3. Restriction Enzyme Map

**What it computes:**
- Cut sites for common enzymes (EcoRI, BamHI, HindIII, etc.)
- Fragment sizes if digested

**Biological meaning:**
- **Cloning strategy** - where to cut/insert
- **Genotyping** - RFLP markers
- **Sequence verification** - expected pattern matches?

**Lab applications:**
```
EcoRI (G^AATTC) cuts at position 45
BamHI (G^GATCC) cuts at position 123
→ Clone insert between these sites
→ Expected fragments: 45bp, 78bp, 200bp
```

#### 4. ORF (Open Reading Frame) Prediction

**What it computes:**
- All ATG start codons
- Stop codons (TAA, TAG, TGA) in same frame
- ORF length (amino acids)

**Biological meaning:**
- **Long ORFs** (>100 aa) = potential protein-coding genes
- **Short ORFs** = regulatory peptides or random
- **Frame shifts** = sequencing errors or real mutations

**Lab applications:**
- Gene discovery in new sequences
- Mutation detection (premature stop codons)
- Synthetic biology (design coding regions)

#### 5. Feature Detection

**What it finds:**
- **TATA box** (TATAAA) - promoter element
- **CpG islands** - methylation regions
- **Poly-A signals** (AATAAA) - transcription termination
- **Start codons** (ATG) - translation start

**Biological meaning:**
- **TATA box** = gene promoter (transcription start)
- **CpG islands** = gene regulation (methylation = silencing)
- **Poly-A signal** = mRNA processing

**Lab applications:**
- Promoter identification
- Epigenetic analysis
- Gene annotation

#### 6. Comparative Analysis

**What it does:**
- Generates random sequence with same composition
- Compares k-mer frequencies
- Reports correlation coefficient

**Biological meaning:**
- **Correlation ~1.0** = sequence looks random (no structure)
- **Correlation <<1.0** = strong non-random patterns (functional)

**Example interpretation:**
```
Correlation: 0.85 → Non-random sequence with regulatory features
Correlation: 0.99 → Mostly random (maybe intergenic region)
Correlation: 0.50 → Highly structured (likely coding or regulatory)
```

### Try It Yourself

**Exercise 1: Analyze a promoter sequence**

```python
from extensions.extension_07_sequence_characterization import SequenceCharacterizer

char = SequenceCharacterizer()

# Typical mammalian promoter
promoter = "TATAAAGGCCACCATGGCTGACTACGTAGCTAG"
report = char.analyze(promoter, name="Sample Promoter")

print(f"GC%: {report['basic_stats']['gc_content']}")
print(f"TATA box: {report['features']['tata_box']}")
print(f"ORFs: {report['orf_analysis']['orf_count']}")
```

**Exercise 2: Compare coding vs non-coding**

```python
# Coding sequence (from a real gene)
coding = "ATGGCTGACTACGTAGCTAGCTAGCTAGCTAGCTAGCTAG"

# Non-coding (random intergenic)
noncoding = "ATATATATATGCGCGCGCGCATATATATATGCGCGC"

coding_report = char.analyze(coding, name="Coding")
noncoding_report = char.analyze(noncoding, name="Non-coding")

# Compare
print(f"Coding ORFs: {coding_report['orf_analysis']['orf_count']}")
print(f"Non-coding ORFs: {noncoding_report['orf_analysis']['orf_count']}")

print(f"Coding k-mer correlation: {coding_report['comparative']['kmer_correlation']}")
print(f"Non-coding k-mer correlation: {noncoding_report['comparative']['kmer_correlation']}")
```

**Expected insight:** Coding sequences typically have:
- Longer ORFs
- Lower k-mer correlation (more structured)
- Codon bias (non-uniform 3-mer frequencies)

**Exercise 3: Design a cloning strategy**

```python
# Your insert sequence
insert = "GAATTCATGGCTGACTACGGATCCGTAGCTAG"

report = char.analyze(insert, name="Cloning Insert")

# Find restriction sites
print("\nRestriction sites for cloning:")
for enzyme, positions in report['restriction_map']['sites'].items():
    if positions:  # Only show enzymes that cut
        print(f"  {enzyme}: positions {positions}")

# Design strategy:
# 1. Cut with enzymes that flank your insert
# 2. Ensure they don't cut inside your insert
# 3. Match with vector polylinker
```

**Exercise 4: Batch analysis**

```python
# Analyze multiple sequences (e.g., from a sequencing run)
sequences = [
    ("Sample_001", "ATGCATGCATGCATGC"),
    ("Sample_002", "GCGCGCGCGCGCGCGC"),
    ("Sample_003", "ATATATATATATATAT"),
]

results = []
for name, seq in sequences:
    report = char.analyze(seq, name=name)
    results.append({
        'name': name,
        'length': report['basic_stats']['length'],
        'gc_content': report['basic_stats']['gc_content'],
        'orf_count': report['orf_analysis']['orf_count'],
    })

# Export to CSV for downstream analysis
import csv
with open('batch_analysis.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'length', 'gc_content', 'orf_count'])
    writer.writeheader()
    writer.writerows(results)
```

### Troubleshooting

**Long sequences (>10kb) are slow:**

The k-mer analysis computes all possible k-mers, which grows exponentially:
- k=1: 4 possibilities
- k=2: 16 possibilities
- k=3: 64 possibilities
- k=4: 256 possibilities

For very long sequences, consider:
```python
# Analyze in chunks
chunk_size = 1000
for i in range(0, len(sequence), chunk_size):
    chunk = sequence[i:i+chunk_size]
    report = char.analyze(chunk, name=f"Chunk_{i}")
```

**No ORFs found:**

This is expected for:
- Short sequences (<30 bp)
- Non-coding regions (promoters, enhancers)
- Sequences without ATG start codons

**Solution:** Check if your sequence is expected to be coding. For promoter analysis, use feature detection instead.

**Restriction map shows too many cuts:**

Some enzymes are **frequent cutters** (4-base recognition):
- AluI (AG^CT) - cuts every ~256 bp on average
- MspI (C^CGG) - cuts every ~256 bp

For cloning, prefer **rare cutters** (6-8 base recognition):
- NotI (GC^GGCCGC) - 8 bases, cuts every ~65kb
- AscI (GG^CGCGCC) - 8 bases

**k-mer correlation always high:**

Short sequences (<50 bp) naturally look random. The correlation is most meaningful for:
- Sequences >100 bp
- Comparing multiple sequences (relative, not absolute)

**TATA box not detected:**

Not all promoters have TATA boxes! Many genes use:
- CpG islands (housekeeping genes)
- Alternative core promoters (Inr, DPE elements)

Absence of TATA box ≠ not a promoter.

### When to Use This vs Agent Demos

| Scenario | Use Demo 07 (Pipeline) | Use Demo 04/05 (Agent) |
|----------|----------------------|-------------------------|
| Analyze 500 sequences | ✅ Batch process | ❌ Too slow/expensive |
| "What's interesting here?" | ❌ Fixed analysis | ✅ Explores adaptively |
| Cloning design | ✅ Restriction map | ❌ May hallucinate sites |
| Paper methods section | ✅ Cite exact algorithm | ❌ LLM = not reproducible |
| Quick one-off question | ❌ Overkill | ✅ Fast answer |
| Privacy-sensitive data | ✅ All local | ❌ API sends data |
| Teaching bioinformatics | ✅ Show real algorithms | ✅ Show LLM capabilities |
| Production pipeline | ✅ Deterministic | ❌ Unreliable |

### What You've Learned

✅ **Comprehensive analysis** - 6 components in one pipeline
✅ **k-mer signatures** - detect non-random patterns
✅ **Restriction mapping** - plan cloning strategies
✅ **ORF prediction** - find potential genes
✅ **Feature detection** - identify regulatory elements
✅ **Comparative analysis** - distinguish functional from random
✅ **Pipeline vs agent trade-offs** - when to skip the LLM

**Key insights:**
1. **Not every problem needs an LLM** - sometimes you just need solid algorithms
2. **Local computation** = fast, private, reproducible
3. **Biological meaning** matters more than raw statistics
4. **Production workflows** benefit from deterministic pipelines
5. **Agents + pipelines** = best of both worlds (demo_06)

**Next steps:**
- → Try your own sequences (promoters, genes, primers)
- → Integrate into your analysis workflow
- → Export results for publications
- → demo_06: Autonomous investigation (agent uses this pipeline!)

---

### 🔍 Detective Mode (Advanced)

**File:** `demo/demo_07b_detective.py`

Want to make sequence analysis more like a **mystery investigation**? The detective mode adds:

| Feature | Characterizer (demo_07) | Detective (demo_07b) |
|---------|------------------------|---------------------|
| **Approach** | Comprehensive pipeline | Hypothesis-driven |
| **API** | None (all local) | NCBI BLAST |
| **Output** | Statistical report | Hypothesis + confidence |
| **Special** | Technical details | Mystery challenges |

**Detective workflow:**
```python
from extensions.extension_05_detective import SequenceDetective

detective = SequenceDetective()

# Mystery sequence challenge
report = detective.investigate("TATAAAAGGCCACCATGGCT", use_cache=False)

print(f"Hypothesis: {report['hypothesis']}")
# Output: "This is likely a promoter region with start codon"
print(f"Confidence: {report['confidence']}")
# Output: 0.85
```

**What the detective does:**
1. Analyzes sequence properties (GC%, motifs)
2. BLAST search against NCBI database
3. Identifies organism/gene from taxonomy
4. Generates hypothesis with confidence score

**Try the mystery challenges:**

```python
MYSTERIES = {
    "Mystery A": "TATAAAAGGCCACCATGGCT",  # Promoter + start codon
    "Mystery B": "AATAAAGCGCGCGCGCGCGC",  # PolyA + CpG island
}

for name, seq in MYSTERIES.items():
    report = detective.investigate(seq)
    print(f"{name}: {report['hypothesis']}")
```

**When to use which:**
- **Characterizer (demo_07)**: Routine analysis, batch processing, publication figures
- **Detective (demo_07b)**: Unknown sequences, teaching challenges, exploration

---

## Demo 06: Hybrid Autonomous Investigation

**File:** `demo/demo_06_hybrid_autonomous.py`

### The Grand Finale: True Agentic Bioinformatics

This is the **culmination** of everything you've learned:

```
demo_01 → Basic LLM calls (what can LLMs do?)
demo_02 → Structured outputs (reliable parsing)
demo_03 → Simple agent (tool orchestration)
demo_04 → Real API agent (external databases)
demo_05 → LLM pattern-matching (EDUCATIONAL ONLY!)
demo_06 → HYBRID (autonomous planning + real tools) ← YOU ARE HERE
```

**Key insight:** demo_06 combines the **best of both worlds**:
- **Autonomy** from agents (LLM plans investigation)
- **Accuracy** from pipelines (real tools do computation)

### The Evolution of Autonomy

| Approach | Planning | Computation | Interpretation | Production Ready? |
|----------|----------|-------------|----------------|-------------------|
| Hardcoded (demo_07) | Human | Python (100%) | Human | ✅ Yes |
| Pure LLM (demo_05) | Human | LLM (pattern-match) | LLM | ❌ Educational only |
| **Hybrid (demo_06)** | **LLM** | **Python (100%)** | **LLM** | ✅ **Yes!** |

**Why hybrid wins:**
- LLM for **what it does well**: planning, reasoning, interpretation
- Python for **what it does well**: computation, accuracy, reproducibility
- **Autonomous + Accurate** = production-ready bioinformatics agent

### What You'll Learn

- **Hybrid agent architecture** - LLM reasoning + real tools
- **Autonomous planning** - agent decides what to analyze
- **Real tool integration** - SequenceCharacterizer from demo_07
- **Biological interpretation** - LLM generates hypotheses
- **Scientific reasoning** - alternative explanations, limitations
- **Production pattern** - how to build real bioinformatics agents

### Run the Demo

```bash
python demo/demo_06_hybrid_autonomous.py
```

### The Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  HUMAN ASKS: "What can you tell me about this sequence?"            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LLM PLANS (Autonomous Decision-Making):                            │
│  "Let me run full characterization first..."                        │
│  "Interesting! GC% is 55%. Let me investigate ORFs..."              │
│  "Found a TATA box! This might be regulatory..."                    │
│  "I have enough to conclude"                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  REAL TOOLS EXECUTE (100% Accurate Computation):                    │
│  → SequenceCharacterizer.analyze()                                  │
│     - GC%: 55.2% (real algorithm)                                   │
│     - ORFs: 3 found (real algorithm)                                │
│     - TATA box: position 0 (real pattern match)                     │
│     - Restriction sites: 5 enzymes (real database lookup)           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LLM INTERPRETS (Biological Reasoning):                             │
│  "This looks like a promoter region because:"                       │
│  "  - TATA box at position 0"                                       │
│  "  - High GC% suggests CpG island"                                 │
│  "  - Short ORF downstream could be regulatory peptide"             │
│  "Confidence: Medium - needs experimental validation"               │
└─────────────────────────────────────────────────────────────────────┘
```

### What Makes It Hybrid (Not Pure LLM)?

**Critical difference from demo_05:**

| Aspect | demo_05 (Pure LLM) | demo_06 (Hybrid) |
|--------|---------------------|-------------------|
| **GC% calculation** | LLM pattern-matches | SequenceCharacterizer (real algorithm) |
| **ORF finding** | LLM guesses | SequenceCharacterizer (real algorithm) |
| **Restriction sites** | LLM hallucinates | SequenceCharacterizer (database lookup) |
| **Confidence** | Hallucinated token | 1.0 for real computation |
| **Production use** | ❌ DANGEROUS | ✅ Safe + Reliable |

**The agent NEVER does calculation itself!**

```python
# ❌ WRONG (demo_05 pattern):
result = llm.call("Calculate GC% of ATGCATGC")
# LLM pattern-matches, might be wrong

# ✅ CORRECT (demo_06 pattern):
plan = llm.call("What should I analyze?")  # LLM reasoning
result = characterizer.analyze(seq)  # Real computation
interpretation = llm.call("What does this mean?")  # LLM reasoning
```

### Agent Workflow

**Example investigation:**

```
Task: "What can you tell me about this sequence?"
Sequence: TATAAAGGCCACCATGGCTGACTACGTAGCTAG

Iteration 1:
  📋 LLM Plan: "Run full characterization first"
  🔬 Real Tools: SequenceCharacterizer.analyze()
     → GC%: 51.52%, ORFs: 1, TATA box: YES
  ✓ Agent: "Interesting findings! Let me investigate further"

Iteration 2:
  📋 LLM Plan: "The TATA box suggests promoter. Check features."
  🔬 Real Tools: Extract from characterization report
     → TATA at position 0, CpG island detected
  ✓ Agent: "This looks regulatory. Let me check ORFs"

Iteration 3:
  📋 LLM Plan: "Analyze the ORF - is it coding?"
  🔬 Real Tools: ORF analysis from report
     → 1 ORF, 33 amino acids (short)
  ✓ Agent: "Short ORF + TATA = likely regulatory peptide"

Iteration 4:
  📋 LLM Plan: "Have enough evidence. Generate conclusion."
  📝 LLM Interpretation: "Promoter region with regulatory peptide"
```

### Report Structure

The agent provides:

- 📋 **Key Findings** - What it discovered (from real tools!)
- 🧬 **Interpretation** - What it means biologically (LLM reasoning)
- 📊 **Evidence** - Data supporting conclusions (real numbers)
- 🤔 **Alternative Explanations** - Scientific skepticism
- ⚠️ **Limitations** - Intellectual honesty
- ❓ **Follow-up Questions** - What to investigate next

### Try It Yourself

**Exercise 1: Promoter identification**

```python
from extensions.extension_06_hybrid_autonomous import AutonomousSequenceAgent

agent = AutonomousSequenceAgent(max_iterations=4)

# Promoter region
seq = "TATAAAGGCCACCATGGCTGACTACGTAGCTAG"
report = agent.investigate(seq, "What type of sequence is this?")

print(report['summary']['biological_interpretation'])
# Expected: "Promoter region" or "Regulatory sequence"
```

**Exercise 2: Coding vs non-coding**

```python
# Long coding sequence
coding_seq = "ATG" + "GCTGACTAC" * 20 + "TAA"  # 183 bp

report = agent.investigate(coding_seq, "Is this coding or non-coding?")

print(f"Confidence: {report['summary']['confidence']}")
print(f"Interpretation: {report['summary']['biological_interpretation']}")
# Expected: "Coding sequence" with high confidence
```

**Exercise 3: Compare hybrid vs pure LLM**

```python
# Run same sequence on both demos
seq = "ATGCATGCATGCATGCATGCATGCATGC"

# demo_05 (pure LLM - educational)
# → LLM pattern-matches GC%, might be wrong

# demo_06 (hybrid - production)
# → SequenceCharacterizer computes GC% (100% accurate)
# → LLM interprets what it means

# Compare results - hybrid should be more reliable!
```

### Troubleshooting

**Agent runs too few iterations:**

Increase `max_iterations`:
```python
agent = AutonomousSequenceAgent(max_iterations=6)  # Deeper investigation
```

**LLM interpretation seems generic:**

Provide more specific task:
```python
# Vague (bad)
task = "Analyze this"

# Specific (good)
task = "Is this a promoter, enhancer, or coding region? Explain your reasoning."
```

**Want to see agent's reasoning:**

Enable `verbose=True`:
```python
report = agent.investigate(seq, task, verbose=True)
# Shows planning, tool calls, and interpretation steps
```

**Confidence seems too high/low:**

Remember: confidence is LLM-generated (like demo_05). The **data** is real, but the confidence estimate is the LLM's guess. Always verify critical findings!

### When to Use Hybrid vs Other Approaches

| Scenario | Use Demo 07 | Use Demo 05 | Use Demo 06 |
|----------|-------------|--------------|--------------|
| Batch analysis (1000s of seqs) | ✅ | ❌ | ❌ |
| Quick calculation | ✅ | ⚠️ | ⚠️ |
| **Exploratory research** | ❌ | ⚠️ | ✅ |
| **Hypothesis generation** | ❌ | ⚠️ | ✅ |
| **Open-ended questions** | ❌ | ⚠️ | ✅ |
| Production pipeline | ✅ | ❌ | ⚠️ |
| **Research assistant** | ❌ | ❌ | ✅ |
| Educational demo | ✅ | ✅ (limitations) | ✅ (best practice) |

**Key insight:** demo_06 is your **AI research partner** - autonomous, insightful, but backed by real data.

### What You've Learned

✅ **Hybrid architecture** - LLM planning + real tools
✅ **Autonomous investigation** - agent decides what to analyze
✅ **Real computation** - SequenceCharacterizer (100% accurate)
✅ **Biological interpretation** - LLM generates hypotheses
✅ **Scientific reasoning** - alternative explanations, limitations
✅ **Production pattern** - how to build real bioinformatics agents

**The grand synthesis:**

1. **demo_01-02**: LLM basics (calls, structured outputs)
2. **demo_03-05**: Agent patterns (orchestration, APIs, pattern-matching)
3. **demo_07**: Production pipelines (no agent, all accuracy)
4. **demo_06**: **HYBRID** (autonomous + accurate = production-ready!)

**This is the power of understanding LLMs:**
- Know what they're bad at → use real tools
- Know what they're good at → use LLM creativity
- Best of both worlds!

**Next steps:**
- → Try both modes on your sequences
- → Add your own creative tools (roasts, raps, memes)
- → Use fun mode for teaching, serious for research
- → Build hybrid agents for your specific domain

---

## Extension Demos

The `extensions/` directory contains reusable components:

| Extension | What It Adds |
|-----------|--------------|
| `extension_01_memory.py` | Conversation persistence |
| `extension_02_batch.py` | Batch processing |
| `extension_03_export.py` | JSON/CSV/Markdown export |
| `extension_04_apis.py` | Real API clients |
| `extension_05_detective.py` | Sequence detective |

### Example: Using Extensions

```python
# Memory - continue conversations across sessions
from extensions.extension_01_memory import MemoryAgent

agent = MemoryAgent(history_file="session.json")
agent.run("Remember I'm studying BRCA1")
# Later...
agent2 = MemoryAgent(history_file="session.json")
agent2.run("What gene was I studying?")  # Remembers!

# Batch - process multiple sequences at once
from extensions.extension_02_batch import BatchAgent

agent = BatchAgent()
agent.run("Analyze: ATGC, GCGCGC, ATATATAT")

# Export - save results
from extensions.extension_03_export import export_all_formats

results = [{"id": "seq1", "gc_content": 50.0}]
export_all_formats(results, "output", title="My Analysis")
# Creates: output.json, output.csv, output.md
```

---

## Learning Path

### For Beginners

Start here:
1. demo_01 → Understand LLM basics
2. demo_02 → Structured outputs
3. demo_03 → See simple agent

### For Intermediate Users

Try these:
1. demo_04 → Real APIs
2. demo_05 → LLM-generated tools
3. demo_06 → Hybrid autonomous agent
4. demo_07 → Local sequence analysis (optional)
5. extensions → Reuse components

### For Advanced Users

Explore:
1. demo_06 → Full autonomy
2. Modify and extend!

---

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

---

## Next Steps

### What You've Learned

✅ Basic LLM API calls (demo_01)
✅ Structured outputs with validation (demo_02)
✅ Simple agent architecture (demo_03)
✅ Real bioinformatics APIs (demo_04)
✅ LLM pattern-matching limitations (demo_05)
✅ Local sequence analysis pipeline (demo_07)
✅ Hybrid autonomous investigation (demo_06)

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

---

## Get Help

- Check existing issues in the repository
- Read the code comments (they're detailed!)
- Experiment and learn by doing

**Happy coding! 🧬🤖**
