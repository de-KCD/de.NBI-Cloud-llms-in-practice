"""
demo_04_agent.py - Real Agentic Workflow for Bioinformatics

This demo shows a simple agentic workflow that connects to actual bioinformatics
databases and APIs.

WHAT MAKES THIS DIFFERENT:

❌ demo_03 (simple agent):
   - Hardcoded toy tools (reverse_complement, count_bases)
   - No real data sources
   - Good for learning basics

✅ demo_04 (real agentic workflow):
   - REAL API tools (UniProt, PubMed, sequence analysis)
   - Multi-step reasoning workflows
   - Agent reflection before acting
   - Error handling and retry logic
   - Conversation memory across iterations

THE AGENT WORKFLOW:

    ┌─────────────────────────────────────────────────────────┐
    │  1. REFLECT: What do I know? What do I need to find?    │
    │  2. PLAN: Which tools will answer remaining questions?  │
    │  3. EXECUTE: Call tools (with error handling)           │
    │  4. SYNTHESIZE: Combine findings into coherent answer   │
    └─────────────────────────────────────────────────────────┘

REAL TOOLS AVAILABLE:
- search_uniprot(): Find protein info from UniProt database
- get_protein_function(): Get detailed function annotations
- search_literature(): Search PubMed for research papers
- translate_dna(): Translate DNA → protein
- find_orfs(): Find open reading frames
- reverse_complement(): Get reverse complement of DNA
- count_bases(): Calculate GC content

EXAMPLE USE CASES:
- "What does the BRCA1 protein do?"
- "Find recent papers on CRISPR and cancer"
- "Translate this sequence and find ORFs"
- "Analyze this gene and find related literature"
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import time
import requests
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
# These are hardcoded for demo/testing purposes only.
#
# In production:
#   API_KEY = os.environ.get("LLM_API_KEY")
#   API_BASE = os.environ.get("LLM_API_BASE")

# Your API key
API_KEY = "your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 8192
TIMEOUT = 300.0


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================
# We use instructor in JSON mode for structured outputs
# This ensures the agent returns valid, parseable responses every time

client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON  # Force JSON - critical for reliability!
)


# =============================================================================
# REAL BIOINFORMATICS API TOOLS
# =============================================================================
# These clients connect to REAL databases used by professional bioinformaticians!
#
# What you'll find here:
# 1. UniProtClient - Protein database (https://uniprot.org)
# 2. LiteratureClient - PubMed literature search
# 3. SequenceAnalysisClient - DNA/protein sequence analysis
#
# Each client:
# - Handles HTTP requests to the API
# - Parses responses into structured data
# - Handles errors gracefully
# - Returns dict with 'success' status
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# UNIPROT CLIENT
# ─────────────────────────────────────────────────────────────────────────────
# UniProt (Universal Protein Resource) is THE database for protein information.
#
# What UniProt provides:
# - Protein sequences
# - Functional annotations
# - Domain information
# - Post-translational modifications
# - Disease associations
# - 3D structure links
#
# API docs: https://www.uniprot.org/help/api_queries
# ─────────────────────────────────────────────────────────────────────────────

class UniProtClient:
    """
    Client for UniProt protein database.

    UniProt is the world's most comprehensive protein database, combining:
    - Swiss-Prot (manually curated, high quality)
    - TrEMBL (automatically annotated)

    Use this client to:
    - Search for proteins by gene name
    - Get protein function annotations
    - Find organism information
    - Retrieve sequence data

    Example:
        >>> client = UniProtClient()
        >>> result = client.search_protein("BRCA1", organism="human")
        >>> print(f"Found: {result['protein_name']}")
        >>> print(f"Length: {result['length']} amino acids")
    """

    # Base URL for UniProt REST API
    BASE_URL = "https://rest.uniprot.org"

    def search_protein(self, gene_name: str, organism: str = "human") -> dict:
        """
        Search UniProt for a protein by gene name.

        This is your gateway to protein information! Give it a gene name
        (like "BRCA1" or "TP53") and get back comprehensive protein data.

        Args:
            gene_name: Gene symbol (e.g., "BRCA1", "TP53", "EGFR")
            organism: Species filter ("human", "mouse", "rat", etc.)
                     Defaults to human for most common use cases

        Returns:
            dict with protein info:
            - success: True/False
            - uniprot_id: UniProt accession (e.g., "P38398" for BRCA1)
            - entry_name: UniProt entry name
            - protein_name: Recommended protein name
            - gene_name: Gene symbol
            - organism: Scientific name
            - length: Protein length (amino acids)

        How UniProt search works:
        1. Build query with gene name and organism filter
        2. Send GET request to /uniprotkb/search endpoint
        3. Parse JSON response
        4. Extract key fields from nested structure

        Example:
            >>> client = UniProtClient()
            >>> result = client.search_protein("BRCA1")
            >>> if result['success']:
            ...     print(f"UniProt ID: {result['uniprot_id']}")
            ...     print(f"Protein: {result['protein_name']}")
            ...     print(f"Length: {result['length']} aa")
        """
        try:
            # ──────────────────────────────────────────────────────────────────
            # STEP 1: Build the search query
            # ──────────────────────────────────────────────────────────────────
            # UniProt uses a specific query syntax:
            # - gene:GENENAME for gene symbol search
            # - organism_id:TAXID for species filter
            # - Taxonomy ID for human = 9606
            #
            # Examples:
            # - "gene:BRCA1 AND organism_id:9606" → human BRCA1
            # - "gene:TP53" → any species (broader search)
            # ──────────────────────────────────────────────────────────────────
            query = f"gene:{gene_name}"

            # Add organism filter if human (most common case)
            if organism.lower() in ["human", "homo sapiens", "9606"]:
                query += " AND organism_id:9606"

            # ──────────────────────────────────────────────────────────────────
            # STEP 2: Send the API request
            # ──────────────────────────────────────────────────────────────────
            # We're using requests.get() for a REST API call
            # The response format is JSON with nested structure
            # ──────────────────────────────────────────────────────────────────
            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": 1  # Just the top match
                },
                timeout=TIMEOUT
            )

            # ──────────────────────────────────────────────────────────────────
            # STEP 3: Handle errors gracefully
            # ──────────────────────────────────────────────────────────────────
            # Sometimes the API might reject our query (400 error)
            # This can happen with certain organism filters
            # Solution: Try again without organism filter as fallback
            # ──────────────────────────────────────────────────────────────────
            if resp.status_code == 400:
                resp = requests.get(
                    f"{self.BASE_URL}/uniprotkb/search",
                    params={
                        "query": f"gene:{gene_name}",
                        "format": "json",
                        "size": 1
                    },
                    timeout=TIMEOUT
                )

            resp.raise_for_status()  # Raise exception for HTTP errors
            data = resp.json()

            # ──────────────────────────────────────────────────────────────────
            # STEP 4: Check if we found anything
            # ──────────────────────────────────────────────────────────────────
            if not data.get("results"):
                return {"success": False, "error": f"Protein '{gene_name}' not found"}

            # ──────────────────────────────────────────────────────────────────
            # STEP 5: Extract the result
            # ──────────────────────────────────────────────────────────────────
            # UniProt responses are deeply nested!
            # We need to navigate through multiple levels safely
            # Using .get() prevents KeyError if field is missing
            # ──────────────────────────────────────────────────────────────────
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
        """
        Get protein function annotations from UniProt.

        This method retrieves detailed functional information about a protein.
        Use this after search_protein() to get deeper insights.

        Args:
            uniprot_id: UniProt accession (e.g., "P38398" for BRCA1)
                       Get this from search_protein() result

        Returns:
            dict with function annotations:
            - success: True/False
            - uniprot_id: The queried ID
            - functions: List of function descriptions (up to 3)
            - protein_name: Recommended protein name

        What you'll get:
        - Molecular function (what the protein does)
        - Biological process (what pathway it's in)
        - Cellular component (where it's located)
        - Disease associations
        - Catalytic activity (if enzyme)

        Example:
            >>> client = UniProtClient()
            >>> result = client.get_protein_function("P38398")
            >>> for func in result['functions']:
            ...     print(f"Function: {func}")
        """
        try:
            # Direct accession lookup - faster than search
            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/{uniprot_id}",
                params={"format": "json"},
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract function comments from nested structure
            # UniProt stores functions in the "comments" array
            # Each comment has a type (FUNCTION, SUBUNIT, etc.)
            functions = []
            for comment in data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    for text in texts:
                        value = text.get("value", "")
                        if value:
                            functions.append(value)  # Truncate long texts

            return {
                "success": True,
                "uniprot_id": uniprot_id,
                "functions": functions if functions else ["No function annotation available"],
                "protein_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# LITERATURE CLIENT
# ─────────────────────────────────────────────────────────────────────────────
# PubMed is the primary database for biomedical literature.
#
# What PubMed provides:
# - 35+ million citations
# - Abstracts and metadata
# - Links to full text
# - Author information
# - Journal details
#
# API: NCBI E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25501/)
# Rate limits: 3 requests/second without API key
# ─────────────────────────────────────────────────────────────────────────────

class LiteratureClient:
    """
    Client for biomedical literature search via PubMed.

    Use this to find research papers about:
    - Specific genes or proteins
    - Diseases or conditions
    - Experimental methods
    - Any biomedical topic

    Example:
        >>> client = LiteratureClient()
        >>> result = client.search_pubmed("BRCA1 AND cancer", max_results=5)
        >>> for paper in result['papers']:
        ...     print(f"{paper['title']}...")
        ...     print(f"  Authors: {paper['authors']}")
        ...     print(f"  Journal: {paper['journal']} ({paper['year']})")
    """

    def search_pubmed(self, query: str, max_results: int = 3) -> dict:
        """
        Search PubMed for relevant papers.

        This uses NCBI's E-utilities API - the official PubMed interface.

        Args:
            query: Search query with support for:
                   - Boolean operators: AND, OR, NOT
                   - Field tags: [Title], [Author], [Journal]
                   - Date ranges: "2020:2024"[Date]
                   Examples:
                   - "BRCA1 AND cancer"
                   - "CRISPR[Title] AND 2023[Date]"
                   - "TP53 AND mutation NOT review"

            max_results: Number of papers to return (default: 3)

        Returns:
            dict with search results:
            - success: True/False
            - query: What you searched for
            - papers: List of paper dicts with:
              - pmid: PubMed ID
              - title: Paper title
              - authors: Author list
              - journal: Journal name
              - year: Publication year

        How PubMed search works (2-step process):
        1. ESEARCH: Find PMIDs matching the query
        2. ESUMMARY: Fetch details for those PMIDs

        Why two steps?
        - ESEARCH returns only IDs (fast)
        - ESUMMARY returns full metadata
        - More efficient than fetching all data upfront

        Example:
            >>> client = LiteratureClient()
            >>> result = client.search_pubmed("p53 tumor suppressor")
            >>> if result['success']:
            ...     print(f"Found {len(result['papers'])} papers")
            ...     for paper in result['papers']:
            ...         print(f"- {paper['title']}...")
        """
        try:
            # ──────────────────────────────────────────────────────────────────
            # STEP 1: ESEARCH - Find matching PMIDs
            # ──────────────────────────────────────────────────────────────────
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_resp = requests.get(
                search_url,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json"
                },
                timeout=TIMEOUT
            )
            search_resp.raise_for_status()

            # Extract list of PMIDs
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return {"success": False, "error": "No papers found"}

            # ──────────────────────────────────────────────────────────────────
            # STEP 2: ESUMMARY - Fetch paper details
            # ──────────────────────────────────────────────────────────────────
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            fetch_resp = requests.get(
                fetch_url,
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),  # Comma-separated list
                    "retmode": "json"
                },
                timeout=TIMEOUT
            )
            fetch_resp.raise_for_status()

            # Extract paper metadata
            papers = []
            for pmid in pmids[:max_results]:
                paper = fetch_resp.json().get("result", {}).get(pmid, {})

                # Handle authors - can be list of dicts or strings
                authors_list = paper.get("authors", [])
                if isinstance(authors_list, list) and len(authors_list) > 0:
                    if isinstance(authors_list[0], dict):
                        author_names = [a.get("name", str(a)) for a in authors_list[:3]]
                    else:
                        author_names = [str(a) for a in authors_list[:3]]
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


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE ANALYSIS CLIENT
# ─────────────────────────────────────────────────────────────────────────────
# Local sequence analysis tools - no API needed!
#
# These are pure Python implementations of common bioinformatics operations:
# - Translation (DNA → protein)
# - ORF finding (identify coding regions)
#
# Why implement locally?
# - Fast (no network latency)
# - Reliable (no API downtime)
# - Educational (see how it works)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceAnalysisClient:
    """
    DNA/protein sequence analysis tools.

    These are local (no API) implementations of common operations.
    Perfect for quick analysis without network calls.
    """

    def translate_dna(self, sequence: str, frame: int = 1) -> dict:
        """
        Translate DNA sequence to protein using the genetic code.

        Translation is the process of converting DNA → protein.
        Every 3 nucleotides (codon) = 1 amino acid.

        Args:
            sequence: DNA sequence (A, T, G, C)
            frame: Reading frame (1, 2, or 3)
                  Frame 1: Start at position 0
                  Frame 2: Start at position 1
                  Frame 3: Start at position 2

        Returns:
            dict with:
            - success: True/False
            - dna_length: Length of input DNA
            - protein_length: Length of protein
            - protein_sequence: Amino acid sequence (single-letter codes)
            - frame: Which frame was used
            - has_stop: Whether stop codon was encountered

        The Genetic Code:
        - 64 codons → 20 amino acids + 3 stop codons
        - ATG = Methionine (start codon)
        - TAA, TAG, TGA = Stop codons (*)
        - Degenerate: multiple codons → same amino acid

        Example:
            >>> client = SequenceAnalysisClient()
            >>> result = client.translate_dna("ATGGCTGACTACGTA")
            >>> print(f"Protein: {result['protein_sequence']}")
            M A D Y V
        """
        # ──────────────────────────────────────────────────────────────────
        # THE GENETIC CODE
        # ──────────────────────────────────────────────────────────────────
        # This dictionary maps each codon (3-letter DNA) to an amino acid.
        # Single-letter codes: A=Ala, R=Arg, N=Asn, D=Asp, C=Cys, etc.
        # * = Stop codon
        # X = Unknown (shouldn't happen with valid DNA)
        # ──────────────────────────────────────────────────────────────────
        genetic_code = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',  # Isoleucine, Methionine
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',  # Threonine
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',  # Asparagine, Lysine
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',  # Serine, Arginine
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',  # Leucine
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',  # Proline
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',  # Histidine, Glutamine
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',  # Arginine
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',  # Valine
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',  # Alanine
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',  # Aspartate, Glutamate
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',  # Glycine
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',  # Tyrosine, Stop
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',  # Serine
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',  # Phenylalanine, Leucine
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',  # Cysteine, Stop, Tryptophan
        }

        try:
            # Clean and normalize sequence
            seq = sequence.upper().replace(" ", "").replace("\n", "")

            # Adjust for reading frame (1-based → 0-based index)
            seq = seq[frame-1:] if frame > 0 else seq

            # Translate codon by codon
            protein = []
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                aa = genetic_code.get(codon, 'X')  # X = unknown
                protein.append(aa)
                if aa == '*':  # Stop codon - end translation
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
        """
        Find open reading frames (ORFs) in DNA sequence.

        An ORF is a potential coding region:
        - Starts with ATG (start codon)
        - Ends with TAA, TAG, or TGA (stop codon)
        - Length is multiple of 3 (codons)

        Args:
            sequence: DNA sequence to analyze
            min_length: Minimum ORF length in nucleotides (default: 30)
                       Shorter ORFs are likely random occurrences

        Returns:
            dict with:
            - success: True/False
            - sequence_length: Length of input
            - orf_count: Number of ORFs found
            - longest_orf: Details of longest ORF
            - orfs: List of top 5 ORFs

        Why ORF finding matters:
        - Identifies potential genes in genomic DNA
        - Predicts protein-coding regions
        - First step in genome annotation
        - Helps distinguish coding vs non-coding DNA

        Example:
            >>> client = SequenceAnalysisClient()
            >>> result = client.find_orfs("ATGGCTGACTACGTAGCTAGCTAG")
            >>> print(f"Found {result['orf_count']} ORFs")
            >>> if result['orfs']:
            ...     orf = result['orfs'][0]
            ...     print(f"Longest: {orf['length']} nt → {orf['protein_len']} aa")
        """
        try:
            seq = sequence.upper().replace(" ", "").replace("\n", "")
            orfs = []

            start_codon = "ATG"
            stop_codons = ["TAA", "TAG", "TGA"]

            # Check all 3 reading frames on forward strand
            for frame in range(3):
                i = frame
                while i < len(seq) - 2:
                    # Look for start codon
                    if seq[i:i+3] == start_codon:
                        # Found start - now look for stop
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
                                break  # Stop looking after first stop codon
                    i += 3

            return {
                "success": True,
                "sequence_length": len(seq),
                "orf_count": len(orfs),
                "longest_orf": max(orfs, key=lambda x: x["length"]) if orfs else None,
                "orfs": orfs[:5]  # Return top 5
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# INITIALIZE CLIENTS
# =============================================================================
# Create singleton instances of each client
# These will be wrapped in tool functions for the agent

uniprot = UniProtClient()
literature = LiteratureClient()
seq_analysis = SequenceAnalysisClient()


# =============================================================================
# TOOL REGISTRY
# =============================================================================
# Tools are the agent's "hands" - how it interacts with the world
#
# Each tool:
# - Has a clear, descriptive name
# - Takes typed arguments
# - Returns dict with 'success' status
# - Handles errors gracefully
#
# The agent learns about these tools from the system prompt
# and decides which ones to call based on the task
# =============================================================================

TOOLS = {
    # UniProt tools - protein database queries
    "search_uniprot": lambda gene, organism="human": uniprot.search_protein(gene, organism),
    "get_protein_function": lambda uniprot_id: uniprot.get_protein_function(uniprot_id),

    # Literature tools - PubMed searches
    "search_literature": lambda query, max_results=3: literature.search_pubmed(query, max_results),

    # Sequence analysis tools - local computation
    "translate_dna": lambda sequence, frame=1: seq_analysis.translate_dna(sequence, frame),
    "find_orfs": lambda sequence, min_length=30: seq_analysis.find_orfs(sequence, min_length),
    "reverse_complement": lambda seq: {"success": True, "result": ''.join({'A':'T','T':'A','G':'C','C':'G'}[b] for b in reversed(seq.upper()))},
    "count_bases": lambda seq: {"success": True, "gc_content": round((seq.upper().count('G') + seq.upper().count('C')) / len(seq) * 100, 2)},
}

# Tool descriptions for the system prompt
# This tells the agent what each tool does
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


# =============================================================================
# AGENT RESPONSE MODELS
# =============================================================================
# These Pydantic models define the structure of agent responses
# instructor ensures the LLM returns valid JSON matching these schemas
# =============================================================================

class ToolCall(BaseModel):
    """
    A tool call with reasoning.

    This model forces the agent to explain WHY it's calling each tool.
    This is crucial for:
    - Debugging agent behavior
    - Understanding agent's thought process
    - Catching mistakes before execution
    """
    tool_name: str = Field(description="Name of tool to call")
    arguments: Dict[str, Any] = Field(description="Tool arguments as key-value pairs")
    reasoning: str = Field(description="Why this tool is being called - what question does it answer?")
    expected_outcome: str = Field(description="What information do you expect to get?")


class AgentReflection(BaseModel):
    """
    Agent's reflection on current state.

    This is the METACOGNITION part of the agent - it thinks about
    its own thinking! Reflection helps the agent:
    - Stay on track
    - Avoid redundant tool calls
    - Know when it has enough information
    - Admit uncertainty
    """
    what_have_we_learned: str = Field(description="Summary of findings so far")
    what_still_unknown: str = Field(description="What questions remain unanswered")
    next_step_rationale: str = Field(description="Why the next action makes sense")
    confidence_in_answer: float = Field(description="Confidence 0-1 in having enough info", ge=0, le=1)
    ready_to_conclude: bool = Field(description="Whether we have enough info to answer")


class AgentResponse(BaseModel):
    """
    Agent's decision at each iteration.

    The agent can:
    1. Reflect on progress
    2. Call one or more tools
    3. Provide final answer (if done)

    The 'done' flag tells us when to stop the loop.
    """
    reflection: Optional[AgentReflection] = Field(default=None, description="Agent's reflection on progress")
    tool_calls: List[ToolCall] = Field(description="Tools to call this iteration")
    final_answer: Optional[str] = Field(default=None, description="Final answer if task is complete")
    done: bool = Field(description="Whether the task is complete")


# =============================================================================
# AGENTIC WORKFLOW CLASS
# =============================================================================
# This is the MAIN EVENT - the agent that runs the show!
#
# What makes this "agentic":
# - REFLECTION: Agent thinks before acting
# - PLANNING: Agent decides which tools to use
# - EXECUTION: Agent calls tools and sees results
# - ITERATION: Agent repeats until satisfied
# - SYNTHESIS: Agent combines findings into answer
#
# This is called a "ReAct" pattern (Reason + Act) in AI research
# =============================================================================

class AgenticWorkflow:
    """
    Real agentic workflow with reflection and multi-step reasoning.

    This agent mimics how a human researcher would investigate a question:

    1. UNDERSTAND the question
    2. PLAN what information is needed
    3. SEARCH for that information (using tools)
    4. EVALUATE what was found
    5. REPEAT until question is answered
    6. SYNTHESIZE findings into coherent answer

    Features:
    - Reflection before each action
    - Error handling and retry logic
    - Memory across iterations
    - Parallel tool execution support
    - Confidence scoring

    Example:
        >>> agent = AgenticWorkflow(max_iterations=8)
        >>> result = agent.run("What does BRCA1 do and what diseases is it associated with?")
        >>> print(result['final_answer'])
    """

    def __init__(self, max_iterations: int = 8, max_retries: int = 2):
        """
        Initialize the agentic workflow.

        Args:
            max_iterations: Maximum tool-calling rounds (default: 8)
                          Most tasks finish in 3-5 iterations
            max_retries: How many times to retry failed tool calls
        """
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.conversation_history = []  # Memory of the conversation
        self.tool_results = []  # Results from tool calls

        # System prompt = agent's "brain" - defines role, tools, and workflow
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
        """
        Execute a tool with error handling and retry logic.

        This is the "muscle" of the agent - it actually DOES things.

        Args:
            tool_name: Name of tool to call (must match TOOLS keys)
            arguments: Dict of arguments to pass to the tool

        Returns:
            dict with tool result (always has 'success' key)

        Error handling strategy:
        1. Try to execute the tool
        2. If it fails, wait 1 second and retry
        3. After max_retries, return error message
        4. Agent sees error and can try different approach
        """
        # Check if tool exists
        if tool_name not in TOOLS:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                result = TOOLS[tool_name](**arguments)

                # Check if tool succeeded
                if isinstance(result, dict) and result.get("success"):
                    return result
                elif attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue

                return result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return {"success": False, "error": f"Tool execution failed: {str(e)}"}

        return {"success": False, "error": "Max retries exceeded"}

    def run(self, task: str, verbose: bool = True) -> dict:
        """
        Run the agentic workflow to complete a task.

        This is the MAIN LOOP - the agent's "central executive".

        Args:
            task: The user's question or request
            verbose: Whether to print progress (default: True)

        Returns:
            dict with:
            - success: True/False
            - final_answer: The agent's answer (if successful)
            - tool_results: All tool results (for debugging)
            - iterations: How many iterations it took

        The loop:
        ┌─────────────────────────────────────────────┐
        │  FOR each iteration (up to max_iterations): │
        │    1. Ask LLM what to do                    │
        │    2. If done=True: return final_answer     │
        │    3. Execute tool calls                    │
        │    4. Add results to conversation history   │
        │    5. Repeat                                │
        └─────────────────────────────────────────────┘
        """
        # Initialize conversation with system prompt + user task
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        self.tool_results = []

        if verbose:
            print(f"\n🔬 Starting agentic workflow for: {task}...")
            print("=" * 60)

        # ──────────────────────────────────────────────────────────────────────
        # MAIN AGENT LOOP
        # ──────────────────────────────────────────────────────────────────────
        response = None
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # ──────────────────────────────────────────────────────────────────
            # STEP 1: Get agent's decision from LLM
            # ──────────────────────────────────────────────────────────────────
            # We send the full conversation history to the LLM.
            # The LLM sees:
            # - System prompt (role, tools, rules)
            # - Original task
            # - All previous tool calls and results
            #
            # The LLM returns an AgentResponse with:
            # - Reflection (what it's thinking)
            # - Tool calls (what it wants to do)
            # - Final answer (if it's done)
            # ──────────────────────────────────────────────────────────────────
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

            # ──────────────────────────────────────────────────────────────────
            # STEP 2: Log reflection (if present)
            # ──────────────────────────────────────────────────────────────────
            # The reflection shows the agent's "inner monologue"
            # This is great for debugging and understanding agent behavior
            # ──────────────────────────────────────────────────────────────────
            if response.reflection and verbose:
                print(f"  🤔 Reflection: {response.reflection.what_still_unknown}...")
                print(f"  📊 Confidence: {response.reflection.confidence_in_answer:.0%}")

            # ──────────────────────────────────────────────────────────────────
            # STEP 3: Check if agent is done
            # ──────────────────────────────────────────────────────────────────
            # If done=True and we have a final_answer, we're finished!
            # The agent has gathered enough information.
            # ──────────────────────────────────────────────────────────────────
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
                    "iterations": iteration + 1
                }

            # ──────────────────────────────────────────────────────────────────
            # STEP 4: Execute tool calls
            # ──────────────────────────────────────────────────────────────────
            # The agent told us which tools to call. Now we execute them!
            # Each tool result is added to the conversation history
            # so the LLM can see the results in the next iteration.
            # ──────────────────────────────────────────────────────────────────
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if verbose:
                        print(f"\n  🔧 Tool: {tool_call.tool_name}")
                        print(f"     Args: {tool_call.arguments}")
                        print(f"     Why: {tool_call.reasoning}...")

                    result = self._execute_tool(tool_call.tool_name, tool_call.arguments)
                    self.tool_results.append({
                        "tool": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "result": result
                    })

                    if verbose:
                        if result.get("success"):
                            print("     ✓ Success")
                        else:
                            print(f"     ✗ Error: {result.get('error', 'Unknown')}")

                # ──────────────────────────────────────────────────────────────
                # STEP 5: Add to conversation history
                # ──────────────────────────────────────────────────────────────
                # We add TWO messages:
                # 1. Assistant's tool call (what it decided to do)
                # 2. User's tool results (what actually happened)
                #
                # This lets the LLM learn from results and decide next steps.
                # ──────────────────────────────────────────────────────────────
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
            else:
                if verbose:
                    print("  ⚠️  No tool calls made")

        # ──────────────────────────────────────────────────────────────────────
        # MAX ITERATIONS REACHED
        # ──────────────────────────────────────────────────────────────────────
        # If we get here, the agent didn't finish in time.
        # This can happen with very complex tasks or if the agent
        # gets stuck in a loop. Return partial results.
        # ──────────────────────────────────────────────────────────────────────
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "partial_answer": response.final_answer if response else None,
            "tool_results": self.tool_results,
            "iterations": self.max_iterations
        }


# =============================================================================
# DEMO MAIN
# =============================================================================
# Let's see the agent in action!
#
# This demo shows the agent tackling realistic bioinformatics questions
# that require multiple steps and tool calls.
#
# Watch how the agent:
# 1. Reflects on what it needs to know
# 2. Plans which tools to use
# 3. Executes tools and sees results
# 4. Synthesizes findings into an answer
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 04: Real Agentic Workflow with Bioinformatics APIs")
    print("=" * 60)
    print()
    print("This demo shows an agent that:")
    print("  ✓ Uses REAL APIs (UniProt, PubMed)")
    print("  ✓ Performs multi-step reasoning")
    print("  ✓ Reflects before acting")
    print("  ✓ Handles errors gracefully")
    print()

    # Example tasks that showcase the agent's capabilities
    demo_tasks = [
        "What does the BRCA1 protein do?",
        "Find recent papers on CRISPR gene editing",
        "Translate ATGGCTGACTACGTA and find ORFs",
    ]

    # Run the agent on the first task (protein function)
    agent = AgenticWorkflow(max_iterations=8)
    result = agent.run(demo_tasks[0], verbose=True)

    # Show summary
    print()
    print("=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Status: {'✓ Success' if result['success'] else '✗ Incomplete'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools called: {len(result['tool_results'])}")

    if result['tool_results']:
        print("\nTool execution trace:")
        for i, tr in enumerate(result['tool_results'], 1):
            status = "✓" if tr['result'].get('success') else "✗"
            print(f"  {i}. {status} {tr['tool']}")

    print()
    print("=" * 60)
    print("DEMO 04 COMPLETE")
    print("=" * 60)
    print()
    print("What you saw:")
    print("  ✓ Real API integration (UniProt, PubMed)")
    print("  ✓ Agent reflection before each action")
    print("  ✓ Multi-step reasoning workflow")
    print("  ✓ Error handling with retry logic")
    print("  ✓ Conversation memory across iterations")
    print()
    print("Try it yourself:")
    print("  → Change the task to analyze your gene of interest")
    print("  → Increase max_iterations for complex queries")
    print("  → Add new tools (see tutorial for guide)")
    print()
    print("Next steps:")
    print("  → demo_05: LLM-generated tools (no hardcoded logic)")
    print("  → demo_06: Fully autonomous investigation")
    print("=" * 60)
