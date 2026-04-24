"""
demo_07_autonomous_research_agent.py - FULL Autonomous Bioinformatics Research

Agent combines THREE tool types:

1. REAL ALGORITHMS (100% deterministic Python):
   - count_bases, find_orfs, search_motif, find_cpg_islands, reverse_complement

2. REAL API CALLS (actual HTTP requests to databases):
   - search_uniprot → UniProt protein database
   - get_protein_function → UniProt function lookup
   - search_pubmed → PubMed literature (with citations)
   - fetch_gene_sequence → NCBI gene database
   - get_protein_sequence → UniProt FASTA

3. LLM REASONING (hypothesis generation, interpretation):
   - interpret_motif → LLM interprets biological significance
   - evaluate_coding_potential → LLM evaluates coding likelihood

KEY INSIGHT:
True research autonomy = LLM orchestrates data retrieval + computation + synthesis

The LLM DECIDES what to analyze, but REAL TOOLS do the actual work.
This is the correct pattern for production bioinformatics agents.

USAGE:
    python demo/demo_07_autonomous_research_agent.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import time
import requests
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime
from collections import Counter
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY="your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
MAX_TOKENS=32768
EXTRA_BODY = {
    "temperature": 0.7,
    "top_p": 0.95,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True
    }
}

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
client = instructor.from_openai(client, mode=instructor.Mode.JSON)

# =============================================================================
# WHY THIS ARCHITECTURE WORKS
# =============================================================================
#
#  +----------------+     +----------------+     +----------------+
#  |   Pre-LLM Era  |  -> |  ChatGPT Era   |  -> |  Agent Era     |
#  +----------------+     +----------------+     +----------------+
#  | Human + Python |     | Human + LLM    |     | LLM + Tools    |
#  |  (manual)      |     |  (prompting)   |     |  (autonomous)  |
#  +----------------+     +----------------+     +----------------+
#
# The trick: LLM is the brain, real tools are the hands.
# LLM cannot calculate - it predicts tokens. But it CAN decide WHAT to calculate.
# Give it real API clients and algorithms, let it orchestrate.
#
# Demo 07 = Demo 04 (real APIs) + Demo 06 (autonomy) + knowledge reasoning
#
# =============================================================================


# =============================================================================
# TOOL CATALOG - MIXED: APIs + Algorithms + Knowledge
# =============================================================================

# --- LOCAL ALGORITHMS ---

def reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(b, 'N') for b in reversed(seq.upper()))


def count_bases(seq: str) -> Dict[str, Any]:
    """Calculate base composition and GC content."""
    seq = seq.upper()
    counts = Counter(seq)
    length = len(seq)
    gc = counts.get('G', 0) + counts.get('C', 0)
    return {
        'length': length,
        'A': counts.get('A', 0),
        'T': counts.get('T', 0),
        'G': counts.get('G', 0),
        'C': counts.get('C', 0),
        'gc_content': round(gc / length * 100, 2) if length > 0 else 0,
    }


def find_orfs(seq: str, min_length: int = 100, genetic_code: str = "standard") -> List[Dict[str, Any]]:
    """Find open reading frames."""
    seq = seq.upper()
    start_codons = {"ATG"}
    stop_codons = {"TAA", "TAG", "TGA"}
    orfs = []

    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            codon = seq[i:i+3]
            if codon in start_codons:
                start = i
                for j in range(i+3, len(seq)-2, 3):
                    if seq[j:j+3] in stop_codons:
                        end = j + 3
                        if end - start >= min_length:
                            protein = translate_dna(seq[start:end])
                            orfs.append({
                                'start': start + 1,
                                'end': end,
                                'strand': '+',
                                'length_nt': end - start,
                                'length_aa': len(protein),
                                'protein': protein[:50] + ('...' if len(protein) > 50 else '')
                            })
                        break
            i += 3

    return sorted(orfs, key=lambda x: x['length_nt'], reverse=True)


def translate_dna(seq: str) -> str:
    """Translate DNA to protein."""
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = codon_table.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def search_motif(seq: str, pattern: str, allow_mismatches: int = 0) -> List[Dict[str, Any]]:
    """Search for motif/pattern in sequence."""
    seq = seq.upper()
    pattern = pattern.upper()
    iupac = {
        'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
        'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
        'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'
    }
    regex_pattern = pattern
    for code, expansion in iupac.items():
        regex_pattern = regex_pattern.replace(code, expansion)

    matches = []
    for match in re.finditer(regex_pattern, seq):
        pos = match.start() + 1
        matches.append({
            'position': pos,
            'matched': match.group(),
            'context': seq[max(0, pos-20):min(len(seq), pos+len(pattern)+20)]
        })
    return matches


def find_cpg_islands(seq: str, min_length: int = 200, min_gc: float = 50.0,
                     min_obs_exp: float = 0.6) -> List[Dict[str, Any]]:
    """Find CpG islands."""
    seq = seq.upper()
    islands = []
    window_size = min_length
    step = 50

    for i in range(0, len(seq) - window_size, step):
        window = seq[i:i+window_size]
        stats = count_bases(window)
        if stats['gc_content'] >= min_gc:
            cpg_count = window.count('CG')
            expected_cpg = (stats['C'] * stats['G']) / len(window)
            obs_exp = cpg_count / expected_cpg if expected_cpg > 0 else 0
            if obs_exp >= min_obs_exp:
                islands.append({
                    'start': i + 1,
                    'end': i + window_size,
                    'gc_content': stats['gc_content'],
                    'cpg_count': cpg_count,
                    'obs_exp_ratio': round(obs_exp, 2)
                })
    return islands


# --- EXTERNAL API CLIENTS ---

class UniProtClient:
    """UniProt protein database client."""

    BASE_URL = "https://rest.uniprot.org"

    def search_protein(self, gene_name: str, organism: str = "human") -> Dict[str, Any]:
        """Search for protein by gene name."""
        try:
            query = f"gene:{gene_name} AND reviewed:true"
            if organism.lower() in ["human", "homo sapiens", "9606"]:
                query += " AND organism_id:9606"

            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/search",
                params={"query": query, "format": "json", "size": 1},
                timeout=30
            )

            if resp.ok and resp.json().get("results"):
                result = resp.json()["results"][0]
                return {
                    "success": True,
                    "uniprot_id": result.get("primaryAccession", "N/A"),
                    "protein_name": result.get("uniProtkbId", "N/A"),
                    "gene_name": gene_name,
                    "function": self._extract_function(result),
                    "length": result.get("sequence", {}).get("length", "N/A")
                }
            return {"success": False, "error": f"Protein not found: {gene_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_function(self, result: dict) -> str:
        """Extract protein function from UniProt entry."""
        comments = result.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    return texts[0].get("value", "N/A")[:200]
        return "Function not available"

    def get_protein_function(self, uniprot_id: str) -> Dict[str, Any]:
        """Get detailed function for a specific UniProt ID."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/uniprotkb/{uniprot_id}",
                params={"format": "json"},
                timeout=30
            )
            if resp.ok:
                result = resp.json()
                return {
                    "success": True,
                    "uniprot_id": uniprot_id,
                    "function": self._extract_function(result),
                    "pathways": self._extract_pathways(result)
                }
            return {"success": False, "error": "Protein not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_pathways(self, result: dict) -> List[str]:
        """Extract pathway information."""
        pathways = []
        comments = result.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "PATHWAY":
                texts = comment.get("texts", [])
                if texts:
                    pathways.append(texts[0].get("value", "")[:100])
        return pathways[:3]


class PubMedClient:
    """PubMed literature search client."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search_articles(self, query: str, max_results: int = 5,
                       year_from: int = None, year_to: int = None) -> Dict[str, Any]:
        """Search PubMed for articles."""
        try:
            # Build query with date filter
            if year_from and year_to:
                query += f" AND ({year_from}:{year_to}[Date - Publication])"

            # Search for IDs
            search_resp = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json"
                },
                timeout=30
            )

            if not search_resp.ok:
                return {"success": False, "error": "PubMed search failed"}

            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return {"success": True, "articles": [], "count": 0}

            # Fetch details
            fetch_resp = requests.get(
                f"{self.BASE_URL}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=30
            )

            articles = []
            if fetch_resp.ok:
                result = fetch_resp.json()
                for pmid in ids:
                    article = result.get("result", {}).get(pmid, {})
                    articles.append({
                        "pmid": pmid,
                        "title": article.get("title", "N/A")[:150],
                        "journal": article.get("fulljournalname", "N/A"),
                        "year": article.get("pubdate", "N/A")[:4],
                        "authors": article.get("authors", [])[:3]
                    })

            return {"success": True, "articles": articles, "count": len(articles)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_abstract(self, pmid: str) -> Dict[str, Any]:
        """Get full abstract for a PubMed ID."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/efetch.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
                timeout=30
            )
            if resp.ok:
                result = resp.json()
                article = result.get("PubmedArticle", [{}])[0].get("MedlineCitation", {}).get("Article", {})
                return {
                    "success": True,
                    "pmid": pmid,
                    "title": article.get("ArticleTitle", "N/A"),
                    "abstract": self._extract_abstract(article)
                }
            return {"success": False, "error": "Abstract not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_abstract(self, article: dict) -> str:
        """Extract abstract text."""
        abstract = article.get("Abstract", {})
        if abstract:
            sections = abstract.get("AbstractText", [])
            if isinstance(sections, list):
                return " ".join(str(s) for s in sections)[:500]
            return str(sections)[:500]
        return "No abstract available"


class NCBISequenceClient:
    """NCBI Nucleotide database client for fetching real sequences."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def fetch_gene_sequence(self, gene_name: str, organism: str = "human",
                            region: str = "promoter") -> Dict[str, Any]:
        """
        Fetch sequence for a gene region from NCBI.

        Args:
            gene_name: Gene symbol (e.g., BRCA1, TP53, TERT)
            organism: Organism name
            region: "promoter" (-2000 to +500 from TSS), "cds" (coding sequence), or "gene"

        Returns:
            Sequence with metadata
        """
        try:
            query = f"{gene_name}[Gene Name] AND {organism}[Organism]"
            search_resp = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={"db": "gene", "term": query, "retmode": "json"},
                timeout=30
            )

            if not search_resp.ok or not search_resp.json().get("esearchresult", {}).get("idlist"):
                return {"success": False, "error": f"Gene not found: {gene_name}"}

            gene_id = search_resp.json()["esearchresult"]["idlist"][0]

            summary_resp = requests.get(
                f"{self.BASE_URL}/esummary.fcgi",
                params={"db": "gene", "id": gene_id, "retmode": "json"},
                timeout=30
            )

            if not summary_resp.ok:
                return {"success": False, "error": "Failed to fetch gene summary"}

            gene_data = summary_resp.json().get("result", {}).get(gene_id, {})
            chrom = gene_data.get("chromosome", "")
            start = gene_data.get("genomicstart", 0)
            end = gene_data.get("genomicend", 0)

            if region == "promoter":
                fetch_start = max(1, start - 2000)
                fetch_end = start + 500
            else:
                fetch_start, fetch_end = start, end

            seq_query = f"{chrom}[Chromosome] AND {fetch_start}:{fetch_end}[Nucleotide Position] AND {organism}[Organism]"
            seq_search = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params={"db": "nucleotide", "term": seq_query, "retmode": "json"},
                timeout=30
            )

            if seq_search.ok and seq_search.json().get("esearchresult", {}).get("idlist"):
                seq_id = seq_search.json()["esearchresult"]["idlist"][0]
                seq_resp = requests.get(
                    f"{self.BASE_URL}/efetch.fcgi",
                    params={"db": "nucleotide", "id": seq_id, "rettype": "fasta", "retmode": "text"},
                    timeout=30
                )

                if seq_resp.ok:
                    lines = seq_resp.text.strip().split("\n")
                    sequence = "".join(line for line in lines[1:] if not line.startswith(">"))
                    return {
                        "success": True,
                        "gene": gene_name,
                        "region": region,
                        "sequence": sequence.upper(),
                        "length": len(sequence),
                        "chromosome": chrom,
                    }

            return {"success": False, "error": "Sequence fetch failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_protein_sequence(self, uniprot_id: str) -> Dict[str, Any]:
        """Fetch protein sequence from UniProt."""
        try:
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=30)
            if resp.ok:
                lines = resp.text.strip().split("\n")
                sequence = "".join(line for line in lines[1:] if not line.startswith(">"))
                return {"success": True, "uniprot_id": uniprot_id, "sequence": sequence, "length": len(sequence)}
            return {"success": False, "error": "Protein not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --- KNOWLEDGE REASONING ---

class KnowledgeEngine:
    """LLM-based reasoning for hypothesis generation (no hardcoded knowledge)."""

    def __init__(self):
        # No hardcoded knowledge - everything goes through LLM
        pass

    def interpret_motif(self, motif: str, position: int, context: str,
                       sequence: str = "") -> Dict[str, Any]:
        """Use LLM to interpret biological significance of a motif."""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=MotifInterpretation,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a bioinformatics expert interpreting DNA motifs.

Provide accurate, evidence-based interpretations. Acknowledge uncertainty.
Distinguish between:
- Well-characterized motifs (TATA box, polyA signal, etc.)
- Putative motifs (matches known pattern but unvalidated)
- Unknown motifs (no known match)"""
                    },
                    {
                        "role": "user",
                        "content": f"""Interpret this DNA motif:

MOTIF: {motif}
POSITION: {position}
CONTEXT: {context}
SEQUENCE: {sequence[:200]}{'...' if len(sequence) > 200 else ''}

What is the biological significance? Is this a known regulatory element?"""
                    }
                ]
            )
            return response.model_dump()
        except Exception as e:
            return {"motif": motif, "position": position, "biological_significance": f"Interpretation failed: {e}", "likely_function": "unknown", "confidence": 0.0}

    def evaluate_coding_potential(self, orfs: List[dict], gc_content: float,
                                  sequence: str = "") -> Dict[str, Any]:
        """Use LLM to evaluate if sequence is likely protein-coding."""
        try:
            orf_summary = f"{len(orfs)} ORF(s) found"
            if orfs:
                longest = max(orfs, key=lambda x: x['length_nt'])
                orf_summary += f", longest: {longest['length_nt']}bp / {longest['length_aa']}aa"

            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=CodingEvaluation,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a bioinformatics expert evaluating coding potential.

Consider:
- ORF length (typical proteins >100aa)
- Start/stop codon presence
- GC content (coding regions often 40-60%)
- Sequence length

Be conservative - many ORFs are false positives."""
                    },
                    {
                        "role": "user",
                        "content": f"""Evaluate coding potential:

{orf_summary}
GC CONTENT: {gc_content}%
SEQUENCE LENGTH: {len(sequence)}bp
SEQUENCE: {sequence[:200]}{'...' if len(sequence) > 200 else ''}

Is this sequence likely protein-coding?"""
                    }
                ]
            )
            return response.model_dump()
        except Exception as e:
            return {"likely_coding": False, "confidence": 0.0, "reasoning": f"Evaluation failed: {e}"}


class MotifInterpretation(BaseModel):
    """LLM interpretation of a DNA motif."""
    motif: str
    position: int
    biological_significance: str = Field(description="What this motif means biologically")
    likely_function: str = Field(description="regulatory, coding, structural, or unknown")
    is_known_motif: bool = Field(description="Is this a well-characterized motif?")
    similar_known_motifs: List[str] = Field(default_factory=list, description="Similar known motifs if any")
    confidence: float = Field(ge=0, le=1, description="Confidence in interpretation")


class CodingEvaluation(BaseModel):
    """LLM evaluation of coding potential."""
    likely_coding: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="Detailed reasoning for the evaluation")
    evidence_for: List[str] = Field(default_factory=list, description="Evidence supporting coding")
    evidence_against: List[str] = Field(default_factory=list, description="Evidence against coding")


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_CATALOG = {
    # Local algorithms
    "count_bases": {
        "type": "algorithm",
        "description": "Calculate base composition and GC content",
        "parameters": {},
        "returns": "length, A/T/G/C counts, gc_content"
    },
    "find_orfs": {
        "type": "algorithm",
        "description": "Find open reading frames (potential coding regions)",
        "parameters": {
            "min_length": {"type": "int", "default": 100, "description": "Minimum ORF length in nucleotides"},
            "genetic_code": {"type": "str", "default": "standard"}
        },
        "returns": "List of ORFs with start, end, strand, length, protein"
    },
    "search_motif": {
        "type": "algorithm",
        "description": "Search for specific DNA motif (TATA box, restriction sites, etc.)",
        "parameters": {
            "pattern": {"type": "str", "required": True, "description": "Motif sequence"},
            "allow_mismatches": {"type": "int", "default": 0}
        },
        "returns": "List of matches with position and context"
    },
    "find_cpg_islands": {
        "type": "algorithm",
        "description": "Find CpG islands (regulatory regions, often near promoters)",
        "parameters": {
            "min_length": {"type": "int", "default": 200},
            "min_gc": {"type": "float", "default": 50.0},
            "min_obs_exp": {"type": "float", "default": 0.6}
        },
        "returns": "List of CpG islands with coordinates"
    },
    "reverse_complement": {
        "type": "algorithm",
        "description": "Get reverse complement of DNA sequence",
        "parameters": {},
        "returns": "Reverse complement sequence"
    },

    # External APIs
    "search_uniprot": {
        "type": "api",
        "description": "Search UniProt protein database by gene name",
        "parameters": {
            "gene_name": {"type": "str", "required": True, "description": "Gene symbol (e.g., BRCA1, TP53)"},
            "organism": {"type": "str", "default": "human"}
        },
        "returns": "Protein info: uniprot_id, name, function, length"
    },
    "get_protein_function": {
        "type": "api",
        "description": "Get detailed protein function from UniProt ID",
        "parameters": {
            "uniprot_id": {"type": "str", "required": True, "description": "UniProt accession (e.g., P38398)"}
        },
        "returns": "Function description and pathways"
    },
    "search_pubmed": {
        "type": "api",
        "description": "Search PubMed literature database",
        "parameters": {
            "query": {"type": "str", "required": True, "description": "Search query"},
            "max_results": {"type": "int", "default": 5},
            "year_from": {"type": "int"},
            "year_to": {"type": "int"}
        },
        "returns": "List of articles with title, journal, year"
    },
    "fetch_gene_sequence": {
        "type": "api",
        "description": "Fetch real gene sequence from NCBI (promoter, CDS, or full gene)",
        "parameters": {
            "gene_name": {"type": "str", "required": True, "description": "Gene symbol (e.g., BRCA1, TP53, TERT)"},
            "organism": {"type": "str", "default": "human"},
            "region": {"type": "str", "default": "promoter", "options": ["promoter", "cds", "gene"]}
        },
        "returns": "DNA sequence with chromosome and coordinates"
    },
    "get_protein_sequence": {
        "type": "api",
        "description": "Fetch protein sequence from UniProt",
        "parameters": {
            "uniprot_id": {"type": "str", "required": True, "description": "UniProt accession (e.g., P38398)"}
        },
        "returns": "Protein sequence"
    },

    # Knowledge reasoning
    "interpret_motif": {
        "type": "knowledge",
        "description": "Interpret biological significance of a DNA motif",
        "parameters": {
            "motif": {"type": "str", "required": True},
            "position": {"type": "int", "required": True},
            "context": {"type": "str", "default": ""}
        },
        "returns": "Biological significance and likely function"
    },
    "evaluate_coding_potential": {
        "type": "knowledge",
        "description": "Evaluate if sequence is likely protein-coding based on ORFs and GC",
        "parameters": {
            "orfs": {"type": "list", "required": True, "description": "List of ORF results"},
            "gc_content": {"type": "float", "required": True}
        },
        "returns": "Coding likelihood with confidence"
    }
}


# =============================================================================
# AGENT SCHEMAS
# =============================================================================

class ToolCall(BaseModel):
    """LLM's decision to run a specific tool."""
    tool_name: str = Field(description="Tool name from catalog")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    reasoning: str = Field(description="Why this tool is needed now")
    hypothesis_being_tested: str = Field(default="", description="What hypothesis this analysis tests")


class ResearchDecision(BaseModel):
    """Agent's decision at each iteration."""
    tool_call: Optional[ToolCall] = Field(description="Tool to run, or None if ready to conclude")
    should_stop: bool = Field(description="True if we have enough to answer the task")
    current_hypothesis: str = Field(description="Working hypothesis based on findings")
    confidence: float = Field(ge=0, le=1, description="Confidence in hypothesis (0-1)")
    alternative_hypotheses: List[str] = Field(default_factory=list, description="Other possible explanations")
    what_i_still_need: str = Field(description="What information is still missing")


class ResearchConclusion(BaseModel):
    """Final research conclusion."""
    answer: str = Field(description="Direct answer to the research question")
    evidence: List[str] = Field(description="Key findings supporting conclusion")
    confidence: float = Field(ge=0, le=1, description="Overall confidence (0-1)")
    hypothesis_summary: str = Field(description="How hypothesis evolved during investigation")
    limitations: str = Field(description="Uncertainties and limitations")
    follow_up_experiments: List[str] = Field(description="Suggested follow-up work")


# =============================================================================
# AUTONOMOUS RESEARCH AGENT
# =============================================================================

# =============================================================================
# AGENT LOOP: THE CORE PATTERN
# =============================================================================
#
# while not_done:
#     1. LLM looks at findings_so_far
#     2. LLM decides: call_tool OR conclude
#     3. If call_tool: execute_tool() -> results go back to LLM
#     4. If conclude: break and generate final answer
#
# Key insight: LLM never computes. It DECIDES what to compute.
# All computation happens in deterministic Python code.
#
# This loop is essentially "chain of thought" made external:
# Instead of LLM thinking in its head (which we cannot see/control),
# we force it to think through tool calls (which we log, cache, inspect).
#
# =============================================================================

class AutonomousResearchAgent:
    """
    Full autonomous research agent combining:
    - External APIs (UniProt, PubMed)
    - Local algorithms (ORFs, motifs, CpG)
    - Knowledge reasoning (hypothesis generation)
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.tool_results = {}
        self.uniprot_client = UniProtClient()
        self.pubmed_client = PubMedClient()
        self.ncbi_client = NCBISequenceClient()
        self.knowledge = KnowledgeEngine()

    def execute_tool(self, tool_name: str, params: Dict[str, Any], seq: str) -> Any:
        """Execute a tool from the catalog."""

        # Local algorithms
        if tool_name == "count_bases":
            return count_bases(seq)
        elif tool_name == "find_orfs":
            return find_orfs(seq, params.get("min_length", 100), params.get("genetic_code", "standard"))
        elif tool_name == "search_motif":
            return search_motif(seq, params.get("pattern", ""), params.get("allow_mismatches", 0))
        elif tool_name == "find_cpg_islands":
            return find_cpg_islands(seq, params.get("min_length", 200),
                                   params.get("min_gc", 50.0), params.get("min_obs_exp", 0.6))
        elif tool_name == "reverse_complement":
            return {"reverse_complement": reverse_complement(seq)}

        # External APIs
        elif tool_name == "search_uniprot":
            return self.uniprot_client.search_protein(
                params.get("gene_name", ""), params.get("organism", "human")
            )
        elif tool_name == "get_protein_function":
            return self.uniprot_client.get_protein_function(params.get("uniprot_id", ""))
        elif tool_name == "search_pubmed":
            return self.pubmed_client.search_articles(
                params.get("query", ""), params.get("max_results", 5),
                params.get("year_from"), params.get("year_to")
            )
        elif tool_name == "fetch_gene_sequence":
            return self.ncbi_client.fetch_gene_sequence(
                params.get("gene_name", ""), params.get("organism", "human"),
                params.get("region", "promoter")
            )
        elif tool_name == "get_protein_sequence":
            return self.ncbi_client.get_protein_sequence(params.get("uniprot_id", ""))

        # Knowledge reasoning
        elif tool_name == "interpret_motif":
            return self.knowledge.interpret_motif(
                params.get("motif", ""), params.get("position", 0),
                params.get("context", ""), seq
            )
        elif tool_name == "evaluate_coding_potential":
            return self.knowledge.evaluate_coding_potential(
                params.get("orfs", []), params.get("gc_content", 0), seq
            )

        return {"error": f"Unknown tool: {tool_name}"}

    def investigate(self, sequence: str, task: str, verbose: bool = True) -> dict:
        """
        Autonomous research investigation.

        Args:
            sequence: DNA sequence (or empty for gene/protein queries)
            task: Research question
            verbose: Print progress

        Returns:
            Research report with findings and conclusion
        """
        seq = sequence.upper().replace(" ", "").replace("\n", "") if sequence else ""

        if verbose:
            print("\n" + "="*60)
            print("AUTONOMOUS RESEARCH AGENT")
            print("="*60)
            print(f"Task: {task}")
            if seq:
                print(f"Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
                print(f"Length: {len(seq)} bp")
            print("\nAvailable tools:")
            print("  Algorithms: count_bases, find_orfs, search_motif, find_cpg_islands")
            print("  APIs: search_uniprot, get_protein_function, search_pubmed, fetch_gene_sequence")
            print("  Knowledge: interpret_motif, evaluate_coding_potential")
            print()

        findings = []

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
                print(f"  Findings: {len(findings)}")

            # LLM decides next action
            decision = self._decide_next_action(seq, task, findings, verbose)

            if decision.should_stop or decision.tool_call is None:
                if verbose:
                    print(f"  Stopping (confidence: {decision.confidence:.0%})")
                break

            # Execute tool
            tool_name = decision.tool_call.tool_name
            params = decision.tool_call.parameters

            if verbose:
                print(f"  Running: {tool_name}({', '.join(f'{k}={v}' for k, v in params.items()) if params else ''})")
                print(f"  Testing: {decision.tool_call.hypothesis_being_tested[:60]}...")

            # Check cache
            cache_key = f"{tool_name}_{json.dumps(params, sort_keys=True)}"
            if cache_key in self.tool_results:
                result = self.tool_results[cache_key]
                if verbose:
                    print(f"  (cached)")
            else:
                result = self.execute_tool(tool_name, params, seq)
                self.tool_results[cache_key] = result

                # If we fetched a sequence, update seq for subsequent analyses
                if tool_name == "fetch_gene_sequence" and result.get("success") and result.get("sequence"):
                    seq = result["sequence"]
                    if verbose:
                        print(f"  → Fetched {result.get('gene')} {result.get('region')} ({result.get('length')}bp)")
                elif TOOL_CATALOG.get(tool_name, {}).get("type") == "api":
                    time.sleep(0.5)

            # Record finding
            findings.append({
                'iteration': iteration + 1,
                'tool': tool_name,
                'tool_type': TOOL_CATALOG.get(tool_name, {}).get("type", "unknown"),
                'parameters': params,
                'result': result,
                'hypothesis_tested': decision.tool_call.hypothesis_being_tested
            })

            if verbose:
                # Detailed output with no truncation
                print(f"  RESULT:")
                if tool_name == "count_bases":
                    print(f"    Length: {result.get('length')}bp")
                    print(f"    A: {result.get('A')}, T: {result.get('T')}, G: {result.get('G')}, C: {result.get('C')}")
                    print(f"    GC%: {result.get('gc_content')}%")
                elif tool_name == "find_orfs":
                    print(f"    Found {len(result)} ORF(s):")
                    for i, orf in enumerate(result[:5], 1):  # Show first 5
                        print(f"      [{i}] {orf['strand']} strand, {orf['length_nt']}bp → {orf['length_aa']}aa")
                        print(f"          Position: {orf['start']}-{orf['end']}")
                        print(f"          Protein: {orf['protein'][:60]}...")
                    if len(result) > 5:
                        print(f"      ... and {len(result) - 5} more")
                elif tool_name == "search_motif":
                    print(f"    Found {len(result)} match(es) for '{params.get('pattern')}':")
                    for i, match in enumerate(result[:5], 1):
                        print(f"      [{i}] Position {match['position']}: {match['matched']}")
                        print(f"          Context: ...{match['context']}...")
                    if len(result) > 5:
                        print(f"      ... and {len(result) - 5} more")
                elif tool_name == "find_cpg_islands":
                    print(f"    Found {len(result)} CpG island(s):")
                    for i, island in enumerate(result[:3], 1):
                        start = island.get('start', 'N/A')
                        end = island.get('end', 'N/A')
                        length = island.get('end', 0) - island.get('start', 0) if island.get('end') and island.get('start') else 'N/A'
                        print(f"      [{i}] {start}-{end} ({length}bp)")
                        print(f"          GC%: {island.get('gc_content', 'N/A')}%, Obs/Exp: {island.get('obs_exp_ratio', 'N/A')}")
                elif tool_name == "search_uniprot":
                    if result.get('success'):
                        print(f"    UniProt ID: {result.get('uniprot_id')}")
                        print(f"    Protein: {result.get('protein_name')}")
                        print(f"    Gene: {result.get('gene_name')}")
                        print(f"    Length: {result.get('length')} aa")
                        print(f"    Function: {result.get('function', 'N/A')[:200]}...")
                    else:
                        print(f"    Error: {result.get('error')}")
                elif tool_name == "get_protein_function":
                    if result.get('success'):
                        print(f"    Function: {result.get('function', 'N/A')[:300]}...")
                        if result.get('pathways'):
                            print(f"    Pathways:")
                            for pw in result['pathways']:
                                print(f"      - {pw}")
                    else:
                        print(f"    Error: {result.get('error')}")
                elif tool_name == "search_pubmed":
                    if result.get('success') and result.get('articles'):
                        print(f"    Found {result['count']} articles:")
                        for i, article in enumerate(result['articles'][:5], 1):
                            print(f"      [{i}] PMID: {article['pmid']}")
                            print(f"          Title: {article['title']}")
                            print(f"          Journal: {article['journal']} ({article['year']})")
                            if article.get('authors'):
                                # Handle both dict and string author formats
                                authors_list = article['authors']
                                if isinstance(authors_list[0], dict):
                                    authors = ', '.join(a.get('name', str(a)) for a in authors_list[:3])
                                else:
                                    authors = ', '.join(str(a) for a in authors_list[:3])
                                print(f"          Authors: {authors}{' et al.' if len(authors_list) > 3 else ''}")
                        if result['count'] > 5:
                            print(f"      ... and {result['count'] - 5} more articles")
                    else:
                        print(f"    Error: {result.get('error', 'No articles found')}")
                elif tool_name == "fetch_gene_sequence":
                    if result.get('success'):
                        print(f"    Gene: {result.get('gene')}")
                        print(f"    Region: {result.get('region')}")
                        print(f"    Chromosome: {result.get('chromosome', 'N/A')}")
                        print(f"    Length: {result.get('length')}bp")
                        print(f"    Sequence (first 100bp): {result.get('sequence', '')[:100]}...")
                    else:
                        print(f"    Error: {result.get('error')}")
                elif tool_name == "get_protein_sequence":
                    if result.get('success'):
                        print(f"    UniProt: {result.get('uniprot_id')}")
                        print(f"    Length: {result.get('length')} aa")
                        print(f"    Sequence: {result.get('sequence', '')[:100]}...")
                elif tool_name == "interpret_motif":
                    print(f"    Motif: {result.get('motif')} at position {result.get('position')}")
                    print(f"    Significance: {result.get('biological_significance', 'N/A')[:200]}...")
                    print(f"    Function: {result.get('likely_function')}")
                    print(f"    Known motif: {result.get('is_known_motif', 'N/A')}")
                    if result.get('similar_known_motifs'):
                        print(f"    Similar to: {', '.join(result['similar_known_motifs'][:3])}")
                    print(f"    Confidence: {result.get('confidence', 0):.0%}")
                elif tool_name == "evaluate_coding_potential":
                    print(f"    Likely coding: {result.get('likely_coding')}")
                    print(f"    Confidence: {result.get('confidence', 0):.0%}")
                    print(f"    Reasoning: {result.get('reasoning', 'N/A')[:200]}...")
                    if result.get('evidence_for'):
                        print(f"    Evidence FOR: {', '.join(result['evidence_for'][:3])}")
                    if result.get('evidence_against'):
                        print(f"    Evidence AGAINST: {', '.join(result['evidence_against'][:3])}")
                else:
                    print(f"    {result}")

        # Generate conclusion
        if verbose:
            print("\n--- Generating Conclusion ---")

        conclusion = self._generate_conclusion(seq, task, findings, verbose)

        report = {
            "timestamp": datetime.now().isoformat(),
            "sequence": seq if seq else None,
            "task": task,
            "iterations": iteration + 1,
            "findings": findings,
            "conclusion": conclusion.model_dump(),
            "tools_used": list(set(f['tool'] for f in findings)),
            "approach": "Autonomous research (APIs + algorithms + knowledge)"
        }

        if verbose:
            print("\n" + "="*60)
            print("RESEARCH COMPLETE")
            print("="*60)
            print(f"\nANSWER:\n{conclusion.answer}\n")
            print(f"EVIDENCE ({len(conclusion.evidence)} items):")
            for i, ev in enumerate(conclusion.evidence, 1):
                print(f"  [{i}] {ev}")
            print(f"\nConfidence: {conclusion.confidence:.0%}")
            print(f"\nHypothesis Evolution:\n{conclusion.hypothesis_summary}\n")
            print(f"Limitations:\n{conclusion.limitations}\n")
            print(f"Follow-up Experiments ({len(conclusion.follow_up_experiments)} suggested):")
            for i, exp in enumerate(conclusion.follow_up_experiments, 1):
                print(f"  [{i}] {exp}")

            # Show PubMed citations if available
            pubmed_findings = [f for f in findings if f['tool'] == 'search_pubmed' and f['result'].get('success')]
            if pubmed_findings:
                print(f"\nPUBMED CITATIONS:")
                for finding in pubmed_findings:
                    articles = finding['result'].get('articles', [])
                    for article in articles:
                        print(f"  • PMID:{article['pmid']} - {article['title'][:80]}...")
                        # Handle author format
                        authors = article.get('authors', [])
                        if authors:
                            if isinstance(authors[0], dict):
                                first_author = authors[0].get('name', 'N/A')
                            else:
                                first_author = str(authors[0])
                            print(f"    {first_author} et al., {article.get('journal', 'N/A')} ({article.get('year', 'N/A')})")
                        else:
                            print(f"    N/A, {article.get('journal', 'N/A')} ({article.get('year', 'N/A')})")
            print()

        return report

    def _decide_next_action(self, seq: str, task: str, findings: List[dict],
                           verbose: bool = True) -> ResearchDecision:
        """LLM decides next research action."""

        # Summarize findings
        findings_summary = []
        for i, f in enumerate(findings, 1):
            tool = f['tool']
            result = f['result']
            if tool == "count_bases":
                findings_summary.append(f"{i}. GC%: {result.get('gc_content')}%")
            elif tool == "find_orfs":
                findings_summary.append(f"{i}. {len(result)} ORF(s), longest: {result[0]['length_aa'] if result else 0}aa")
            elif tool == "search_motif":
                findings_summary.append(f"{i}. Found {len(result)} {f['parameters'].get('pattern')} match(es)")
            elif tool == "search_uniprot":
                if result.get('success'):
                    findings_summary.append(f"{i}. UniProt: {result.get('protein_name')} - {result.get('function', 'N/A')[:50]}")
                else:
                    findings_summary.append(f"{i}. UniProt: {result.get('error')}")
            elif tool == "search_pubmed":
                findings_summary.append(f"{i}. PubMed: {result.get('count')} articles found")
            else:
                findings_summary.append(f"{i}. {tool}: {str(result)[:60]}")

        findings_text = "\n".join(findings_summary) if findings_summary else "No findings yet."

        # Build tool catalog
        tool_descriptions = []
        for name, info in TOOL_CATALOG.items():
            params_str = ", ".join(f"{k}={v.get('default', 'REQUIRED')}"
                                   for k, v in info['parameters'].items()) if info['parameters'] else ""
            tool_descriptions.append(f"- {name}({params_str}) [{info['type']}]: {info['description']}")
        tools_text = "\n".join(tool_descriptions)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=ResearchDecision,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an autonomous bioinformatics research agent.

AVAILABLE TOOLS:
{tools_text}

TOOL TYPES:
- algorithm: Local computation (fast, 100% accurate)
- api: External database query (UniProt, PubMed)
- knowledge: Reasoning/hypothesis generation

RULES:
1. Choose ONE tool per iteration with specific parameters
2. Explain what hypothesis you're testing
3. Track alternative hypotheses
4. Stop when confident (>0.8) or max iterations reached
5. For sequence analysis: start with basics, then dive deeper
6. For gene/protein queries: search databases, then literature"""
                    },
                    {
                        "role": "user",
                        "content": f"""RESEARCH TASK: {task}

SEQUENCE: {len(seq)}bp{' - ' + seq[:50] + '...' if seq else ' (none provided)'}

FINDINGS SO FAR:
{findings_text}

What should I investigate next? If ready to conclude, set should_stop=True."""
                    }
                ]
            )

            if verbose:
                if response.tool_call:
                    print(f"  Decision: {response.tool_call.tool_name}")
                else:
                    print(f"  Decision: Conclude ({response.confidence:.0%})")
                print(f"  Hypothesis: {response.current_hypothesis[:70]}...")

            return response

        except Exception as e:
            if verbose:
                print(f"  Decision failed: {e}")
            return ResearchDecision(
                tool_call=ToolCall(
                    tool_name="count_bases",
                    parameters={},
                    reasoning="Default: get basic stats",
                    hypothesis_being_tested="Sequence composition analysis"
                ),
                should_stop=False,
                current_hypothesis="Analysis in progress",
                confidence=0.3,
                alternative_hypotheses=[],
                what_i_still_need="Basic sequence statistics"
            )

    def _generate_conclusion(self, seq: str, task: str, findings: List[dict],
                            verbose: bool = True) -> ResearchConclusion:
        """Generate final research conclusion."""

        findings_summary = []
        for f in findings:
            findings_summary.append(f"- {f['tool']} ({f['tool_type']}): {str(f['result'])[:100]}")
        findings_text = "\n".join(findings_summary)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=ResearchConclusion,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a bioinformatics research expert.

Synthesize findings into a clear conclusion:
- State the answer directly
- List supporting evidence
- Acknowledge limitations
- Suggest follow-up experiments

Be scientifically rigorous and honest about uncertainty."""
                    },
                    {
                        "role": "user",
                        "content": f"""RESEARCH TASK: {task}

SEQUENCE: {len(seq)}bp{' - ' + seq[:50] if seq else ' (none)'}

INVESTIGATION FINDINGS:
{findings_text}

Provide your research conclusion."""
                    }
                ]
            )

            if verbose:
                print(f"  Conclusion generated")

            return response

        except Exception as e:
            if verbose:
                print(f"  Conclusion failed: {e}")
            return ResearchConclusion(
                answer="Analysis incomplete",
                evidence=["Investigation did not complete"],
                confidence=0.1,
                hypothesis_summary="Could not generate conclusion",
                limitations="Analysis failed",
                follow_up_experiments=["Re-run analysis"]
            )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DEMO 07: AUTONOMOUS RESEARCH AGENT")
    print("="*60)
    print("\nTOOL TYPES:")
    print("  ✓ REAL ALGORITHMS: count_bases, find_orfs, search_motif (100% deterministic)")
    print("  ✓ REAL API CALLS: UniProt, PubMed, NCBI (actual HTTP requests)")
    print("  ✓ LLM REASONING: interpret_motif, evaluate_coding_potential")
    print("\nThe LLM DECIDES what tools to run - REAL TOOLS do the work.")
    print("This is the correct pattern for production bioinformatics agents.\n")

    agent = AutonomousResearchAgent(max_iterations=8)

    # Example 1: Fetch and analyze real TERT promoter sequence
    print("\n" + "="*60)
    print("EXAMPLE 1: Fetch and analyze real TERT promoter")
    print("="*60)

    report1 = agent.investigate(
        "",  # No sequence provided - agent will fetch it
        "Fetch the TERT gene promoter sequence and analyze it. What regulatory elements does it contain? Is it associated with cancer?",
        verbose=True
    )

    # Example 2: BRCA1 gene function
    print("\n" + "="*60)
    print("EXAMPLE 2: BRCA1 gene function and disease association")
    print("="*60)

    agent2 = AutonomousResearchAgent(max_iterations=6)
    report2 = agent2.investigate(
        "",
        "What is the function of BRCA1 protein and what diseases is it associated with? Fetch the sequence and analyze it.",
        verbose=True
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nThis demo shows full research autonomy:")
    print("  ✓ Agent chooses between APIs, algorithms, and reasoning")
    print("  ✓ Agent tracks hypotheses being tested")
    print("  ✓ Agent synthesizes findings into conclusions")
    print("  ✓ Agent suggests follow-up experiments")
    print("\nCompare to other demos:")
    print("  - demo_03: Simple agent with hardcoded local tools")
    print("  - demo_04: Agent with external API tools only")
    print("  - demo_05: Agent with LLM knowledge only")
    print("  - demo_07: FULL agent - APIs + algorithms + knowledge")
    print("="*60)
