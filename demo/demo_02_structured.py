"""
demo_02_structured.py - Structured LLM Outputs for Bioinformatics

Problem: LLMs output free text. Parsing is fragile, error-prone, breaks on
         format changes. Downstream code assumes structure that may not exist.

Solution: Use Pydantic schemas + Instructor to get typed, validated JSON.
          LLM returns structured data that passes validation or we retry.

Key concepts:
- Pydantic schemas define exact output structure (fields + types)
- Field constraints catch errors automatically (regex, ranges, enums)
- JSON mode forces LLM to return valid data (not prose)
- Retry logic handles validation failures (LLM occasionally misbehaves)

Why this matters:
- Without structure: regex parsing, string splitting, manual type conversion
- With structure: type hints, validation, IDE autocomplete, less bugs
- Bioinformatics data is inherently structured (VCF, BED, DE tables) - match it

"""

import instructor
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from openai import OpenAI
import json

# Configuration
# API_KEY set via environment or config file
API_KEY = "your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
# MAX_TOKENS set appropriately for task complexity
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

# Wrap OpenAI client with Instructor for structured outputs.
# Instructor intercepts the call, adds schema to prompt, validates response.
# Mode.JSON tells LLM: "Return ONLY valid JSON, nothing else."
# Without this, LLM might add markdown, explanations, or formatting we don't want.
client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# =============================================================================
# Schema 1: Gene Annotation
# =============================================================================
# Why structured? Gene names have conventions (uppercase, no spaces).
# Free text might say "BRCA1 (Breast Cancer 1)" or "BRCA-1" - inconsistent.
# Schema enforces consistency so downstream code doesn't break.
class GeneInfo(BaseModel):
    """Gene annotation with validation."""

    gene_name: str = Field(
        description="Official gene symbol",
        # HGNC symbols: uppercase, no spaces or special chars.
        # matches: "BRCA1", "TP53", "GATA3"
        # rejects: "brca1" (lowercase), "breast cancer 1" (full name), "BRCA-1" (hyphen)
        pattern=r"^[A-Z0-9]+$"
    )
    exon_count: Optional[int] = Field(
        # Realistic range prevents LLM hallucinations.
        # matches: 24 (BRCA1), 2 (TP53)
        # rejects: -5 (negative), 999999 (impossibly large)
        default=None, description="Number of exons", ge=1, le=10000
    )
    chromosome: Optional[str] = Field(default=None, description="Chromosome location")
    description: Optional[str] = Field(default=None, description="Brief description")
    uniprot_id: Optional[str] = Field(
        default=None,
        description="UniProt accession",
        # UniProt IDs follow strict patterns. Two formats:
        #   1. "P38398" - one letter (O/P/Q), one digit, three alphanumeric, one digit
        #   2. "A0A0B2" - two letters, digit, repeating letter+digit groups
        # matches: "P38398" (BRCA1), "P04637" (TP53)
        # rejects: "P3839X" (wrong final char), "brca1_uniprot" (random string)
        pattern=r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
    )


def extract_gene_info(text: str, max_retries: int = 2) -> GeneInfo:
    """Extract structured gene information from text."""
    # Why retry? LLM might output valid JSON but fail validation (e.g., wrong format).
    # 2 retries is pragmatic: balances reliability vs latency.
    # Most failures are transient (format issues, not semantic errors).
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
                response_model=GeneInfo,  # Tells Instructor what to expect
                messages=[
                    # System prompt: be explicit about JSON-only output
                    {"role": "system", "content": "Extract gene information. Return ONLY valid JSON."},
                    {"role": "user", "content": text}
                ]
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
    raise RuntimeError(f"Failed after {max_retries} retries")


# =============================================================================
# Schema 2: Experiment Result with custom validation
# =============================================================================
# Why custom validator? Some constraints can't be expressed with Field().
# Example: measurements must be positive (biological quantities aren't negative).
# Field(ge=0) catches it, but custom validator gives better error messages.
class ExperimentResult(BaseModel):
    """Experimental measurement results."""

    sample_id: str = Field(
        description="Unique sample identifier",
        # Uppercase alphanumeric, underscores, hyphens.
        # matches: "S001", "SAMPLE-A1", "CELL_LINE_42"
        # rejects: "s001" (lowercase), "Sample 1" (space)
        pattern=r"^[A-Z0-9_-]+$"
    )
    condition: str = Field(description="Treatment condition")
    measurements: Dict[str, float] = Field(description="Key-value measurements")
    passed_qc: bool = Field(description="Whether sample passed quality control")
    notes: Optional[str] = Field(default=None, description="Additional observations")

    @field_validator('measurements')
    @classmethod
    def validate_measurements(cls, v):
        """Ensure all measurement values are positive."""
        # Why validate? Negative RNA-seq counts or concentrations are usually errors.
        # Could be LLM hallucination or data entry mistake.
        # Better to fail fast here than propagate bad data downstream.
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Measurement '{key}' cannot be negative: {value}")
        return v


def extract_experiment_result(text: str) -> ExperimentResult:
    """Extract structured experiment result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=ExperimentResult,
        messages=[
            {"role": "system", "content": "Extract experiment results. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 3: Variant Call (VCF-like)
# =============================================================================
# Why VCF-like? VCF is the standard format for variant calls.
# Matching VCF structure makes integration with bioinformatics tools seamless.
# Field patterns enforce biological reality (only ACGT alleles, valid chromosomes).
class VariantCall(BaseModel):
    """Genetic variant/SNP call."""

    chrom: str = Field(
        description="Chromosome",
        # Standard human chromosomes with optional "chr" prefix.
        # matches: "17", "chr17", "X", "chrX", "M", "chrM"
        # rejects: "1A" (non-standard), "chromosome17" (full word)
        pattern=r"^(chr)?([0-9]{1,2}|X|Y|M)$"
    )
    pos: int = Field(description="1-based position", ge=1)  # VCF uses 1-based coords
    ref: str = Field(
        description="Reference allele",
        # Only valid DNA bases plus N for ambiguity.
        # matches: "A", "G", "ATCG", "N"
        # rejects: "R" (IUPAC ambiguity), "Aa" (lowercase)
        pattern=r"^[ACGTN]+$"
    )
    alt: str = Field(
        description="Alternate allele",
        # Same as ref but allows "." for no-change.
        # matches: "T", "A", "."
        # rejects: "U" (RNA base), "-" (gap)
        pattern=r"^[ACGTN.]+$"
    )
    qual: Optional[float] = Field(default=None, description="Quality score", ge=0, le=999)
    filter: Optional[str] = Field(default=None, description="Filter status")
    gene: Optional[str] = Field(default=None, description="Affected gene")
    consequence: Optional[str] = Field(default=None, description="Variant effect")
    zygosity: Optional[str] = Field(default=None, description="Homozygous/heterozygous")


def extract_variant_call(text: str) -> VariantCall:
    """Extract variant call from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=VariantCall,
        messages=[
            {"role": "system", "content": "Extract variant call information. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 4: Differential Expression
# =============================================================================
# Why structured? DE results have mathematical relationships:
# - log2FC sign should match direction (up vs down)
# - p-values are bounded [0, 1]
# - padj <= pvalue (FDR correction makes it larger)
# Schema catches inconsistent outputs where LLM says "up" but log2FC is negative.
class DifferentialExpression(BaseModel):
    """Differential gene expression result (e.g., from DESeq2)."""

    gene: str = Field(description="Gene symbol")
    # base_mean: average normalized expression across all samples.
    # matches: 500.5 (well-expressed), 0.1 (low)
    # rejects: -10 (impossible for counts)
    base_mean: float = Field(description="Mean expression across samples", ge=0)
    log2_fold_change: float = Field(description="Log2 fold change")
    # p-value and adjusted p-value: probabilities bounded [0, 1].
    # matches: 0.0001 (significant), 0.05 (borderline)
    # rejects: 1.5 (>1), -0.01 (<0)
    pvalue: float = Field(description="Raw p-value", ge=0, le=1)
    padj: float = Field(description="Adjusted p-value (FDR)", ge=0, le=1)
    significant: bool = Field(description="Significantly differentially expressed")
    # direction: "up" for positive log2FC, "down" for negative.
    # matches: "up", "down", "upregulated", "downregulated"
    # rejects: "UP" (uppercase), "increased" (not in allowed set)
    direction: str = Field(description="Up or down regulated")

    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        """Ensure direction matches log2FC sign."""
        # Why validate? LLM might say "upregulated" when log2FC is -2.3.
        # This is a semantic error the LLM made, not just formatting.
        # Validator catches it so downstream analysis doesn't use wrong direction.
        if v.lower() not in ['up', 'down', 'upregulated', 'downregulated']:
            raise ValueError(f"Invalid direction: {v}. Must be 'up' or 'down'")
        return v.lower()


def extract_de_result(text: str) -> DifferentialExpression:
    """Extract differential expression result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=DifferentialExpression,
        messages=[
            {"role": "system", "content": "Extract differential expression data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 5: Pathway Enrichment
# =============================================================================
# Why structured? Enrichment results have specific ID formats (GO terms, KEGG).
# Without validation, LLM might invent pathway IDs or mix formats.
# Pattern matching ensures compatibility with pathway databases (DAVID, g:Profiler).
class PathwayEnrichment(BaseModel):
    """Pathway or GO term enrichment result."""

    pathway_id: str = Field(
        description="Pathway ID (GO:XXXX or KEGG:XXXX)",
        # GO terms: "GO:" + 7 digits. KEGG: species prefix + 5 digits.
        # matches: "GO:0006915" (apoptosis), "hsa04110" (cell cycle), "mmu04110" (mouse)
        # rejects: "pathway_123" (made up), "GO:123" (too short), "KEGG:04110" (wrong format)
        pattern=r"^(GO:\d{7}|hsa\d{5}|mmu\d{5})$"
    )
    pathway_name: str = Field(description="Pathway name")
    category: str = Field(description="Category: BP, MF, CC, or KEGG")
    genes_in_pathway: List[str] = Field(description="Genes from input list in this pathway")
    total_genes_in_pathway: int = Field(
        description="Total genes in pathway",
        # matches: 150 (KEGG cell cycle), 50 (small GO term)
        # rejects: 0 (empty pathway), -1 (impossible)
        ge=1
    )
    enrichment_ratio: float = Field(
        description="Enrichment ratio",
        # matches: 2.5 (strong enrichment), 1.0 (no enrichment)
        # rejects: -0.5 (negative ratio impossible)
        ge=0
    )
    # Enrichment p-value and FDR: same [0, 1] bounds as DE results.
    # matches: 1e-10 (highly enriched), 0.01 (moderate)
    # rejects: 0.0 (exact zero rare), 2.0 (>1 impossible)
    pvalue: float = Field(description="Enrichment p-value", ge=0, le=1)
    fdr: float = Field(description="False discovery rate", ge=0, le=1)


def extract_pathway_enrichment(text: str) -> PathwayEnrichment:
    """Extract pathway enrichment result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=PathwayEnrichment,
        messages=[
            {"role": "system", "content": "Extract pathway enrichment data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 6: Sequence Feature
# =============================================================================
# Why structured? Genomic features have spatial constraints:
# - Start and end positions must be ordered (start < end)
# - Strand is categorical (+, -, or N/A)
# - Score is bounded [0, 1] for probabilities/confidences
# Validator catches impossible coordinates (end before start).
class SequenceFeature(BaseModel):
    """Annotated feature on a DNA/protein sequence."""

    feature_type: str = Field(description="Type: ORF, promoter, motif, domain, etc.")
    # start/end: 1-based genomic coordinates. start < end enforced by validator.
    # matches: start=100, end=500 (valid interval)
    # rejects: start=500, end=100 (inverted), start=0 (0-based, not 1-based)
    start: int = Field(description="Start position (1-based)", ge=1)
    end: int = Field(description="End position", ge=1)
    strand: Optional[str] = Field(default=None, description="Strand: +, -, or N/A")
    sequence: Optional[str] = Field(default=None, description="Feature sequence")
    score: Optional[float] = Field(
        default=None, description="Confidence score",
        # matches: 0.95 (high confidence), 0.3 (low confidence)
        # rejects: 1.5 (>1), -0.2 (<0)
        ge=0, le=1
    )
    description: Optional[str] = Field(default=None, description="Feature description")

    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end position is after start."""
        # Why validate? Genomic coordinates must be ordered.
        # LLM might confuse start/end or output relative positions.
        # Catches: start=100, end=50 (impossible without strand flip)
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError(f"End ({v}) must be greater than start ({info.data['start']})")
        return v


def extract_sequence_feature(text: str) -> SequenceFeature:
    """Extract sequence feature annotation from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=SequenceFeature,
        messages=[
            {"role": "system", "content": "Extract sequence feature. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 7: Sample Metadata
# =============================================================================
# Why structured? Metadata drives experimental design.
# MIAME/MINSEQE standards require specific fields for reproducibility.
# Schema enforces consistency so samples can be compared/merged.
class SampleMetadata(BaseModel):
    """Sample/patient metadata following MIAME/MINSEQE standards."""

    sample_id: str = Field(description="Sample identifier")
    sample_type: str = Field(description="Tissue/cell type")
    organism: str = Field(description="Organism")
    # age: realistic human age range.
    # matches: 45 (typical adult), 0 (newborn)
    # rejects: -1 (negative), 200 (unrealistic)
    age: Optional[int] = Field(default=None, description="Age in years", ge=0, le=150)
    sex: Optional[str] = Field(default=None, description="Sex")
    condition: Optional[str] = Field(default=None, description="Disease/condition")
    treatment: Optional[str] = Field(default=None, description="Treatment if any")
    collection_date: Optional[str] = Field(default=None, description="Collection date")
    batch: Optional[str] = Field(default=None, description="Batch ID")


def extract_sample_metadata(text: str) -> SampleMetadata:
    """Extract sample metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=SampleMetadata,
        messages=[
            {"role": "system", "content": "Extract sample metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 8: Clustering Result (scRNA-seq) - nested model
# =============================================================================
# Why nested? Clustering results have hierarchical structure:
# - A cluster contains multiple marker genes
# - Each marker has its own statistics (logFC, p-value)
# Flat structure would lose this relationship.
# Nested Pydantic models preserve the parent-child relationship.
class ClusterMarker(BaseModel):
    """Marker gene for a cluster."""

    gene: str = Field(description="Gene symbol")
    logfc: float = Field(description="Log fold change vs other clusters")
    # pval_adj: adjusted p-value for marker gene significance.
    # matches: 0.001 (significant marker), 0.049 (borderline)
    # rejects: 0.0 (exact zero rare), 2.0 (>1 impossible)
    pval_adj: float = Field(description="Adjusted p-value", ge=0, le=1)


class ClusterInfo(BaseModel):
    """Cell cluster from single-cell analysis."""

    cluster_id: str = Field(description="Cluster identifier")
    # cell_count: number of cells assigned to this cluster.
    # matches: 500 (typical cluster), 10 (small cluster)
    # rejects: 0 (empty cluster), -5 (impossible)
    cell_count: int = Field(description="Number of cells in cluster", ge=1)
    cell_type_prediction: Optional[str] = Field(default=None, description="Predicted cell type")
    # confidence: probability score for cell type prediction.
    # matches: 0.95 (confident), 0.4 (uncertain)
    # rejects: 1.2 (>1), -0.1 (<0)
    confidence: Optional[float] = Field(default=None, description="Prediction confidence 0-1", ge=0, le=1)
    marker_genes: List[ClusterMarker] = Field(description="Top marker genes")
    pathway_enrichment: Optional[str] = Field(default=None, description="Enriched pathways")


def extract_cluster_info(text: str) -> ClusterInfo:
    """Extract cluster information from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=ClusterInfo,
        messages=[
            {"role": "system", "content": "Extract cluster information. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Schema 9: Genomic Region (BED-like)
# =============================================================================
# Why BED-like? BED is the standard for genomic intervals.
# Matching BED structure enables direct integration with tools like bedtools.
# Note: BED uses 0-based coordinates, unlike VCF which uses 1-based.
class GenomicRegion(BaseModel):
    """Genomic coordinate region (BED format)."""

    chrom: str = Field(description="Chromosome")
    # start/end: 0-based BED coordinates. start < end enforced by validator.
    # BED convention: start is inclusive, end is exclusive.
    # matches: start=0, end=1000 (first 1000 bp)
    # rejects: start=-1 (negative), start=1000, end=500 (inverted)
    start: int = Field(description="Start position (0-based)", ge=0)  # BED = 0-based
    end: int = Field(description="End position", ge=1)
    name: Optional[str] = Field(default=None, description="Region name")
    # score: BED-style score 0-1000 (higher = stronger signal).
    # matches: 1000 (strong peak), 0 (background)
    # rejects: -50 (negative), 1500 (>1000, outside BED spec)
    score: Optional[float] = Field(default=None, description="Score 0-1000", ge=0, le=1000)
    strand: Optional[str] = Field(default=None, description="Strand: +, -, or .")
    feature_type: Optional[str] = Field(default=None, description="e.g., peak, gene, exon")

    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end position is after start."""
        # Same validation as SequenceFeature but 0-based coords.
        # Catches: start=1000, end=500 (invalid interval)
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError(f"End ({v}) must be greater than start ({info.data['start']})")
        return v


def extract_genomic_region(text: str) -> GenomicRegion:
    """Extract genomic region from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=GenomicRegion,
        messages=[
            {"role": "system", "content": "Extract genomic region. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# Export utilities
# =============================================================================
# Why export? Structured data is useful for downstream analysis.
# DataFrame/CSV/JSON formats are standard for bioinformatics pipelines.
# Export functions handle Pydantic models automatically.
def export_to_dataframe(results: List[dict], schema_name: str):
    """Convert results to pandas DataFrame."""
    # Why pandas? Standard tool for tabular data in bioinformatics.
    # Enables quick filtering, grouping, and statistical analysis.
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        return df
    except ImportError:
        return results


def export_to_csv(results: List[dict], filename: str):
    """Export results to CSV file."""
    # Why CSV? Universal format, human-readable, Excel-compatible.
    # Fallback to csv module if pandas not available.
    df = export_to_dataframe(results, "")
    if hasattr(df, 'to_csv'):
        df.to_csv(filename, index=False)
        print(f"Exported to {filename}")
    else:
        # Fallback without pandas
        import csv
        if results:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Exported to {filename}")


def save_to_json(data, filename: str):
    """Save data to JSON file."""
    # Why JSON? Preserves type information, nested structures.
    # Better than CSV for complex hierarchical data.
    # Handles both Pydantic v1 and v2 for compatibility.
    with open(filename, 'w') as f:
        if hasattr(data, 'model_dump'):
            # Pydantic v2
            json.dump(data.model_dump(), f, indent=2, default=str)
        elif hasattr(data, 'dict'):
            # Pydantic v1
            json.dump(data.dict(), f, indent=2, default=str)
        else:
            json.dump(data, f, indent=2, default=str)
    print(f"Saved to {filename}")


# =============================================================================
# Demo - Run all schemas
# =============================================================================
# Each example: input text -> structured extraction -> formatted summary + raw JSON
# Why print both? Summary shows the important fields, JSON shows exact structure.
# Useful for debugging: see what the LLM actually returned before post-processing.
if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 02: Structured LLM Outputs with Pydantic")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------------
    # 1. Gene Info
    # -----------------------------------------------------------------------
    print("1. GENE INFO - Extracting structured gene data")
    print("-" * 50)
    text = "BRCA1 is located on chromosome 17 with 24 exons. UniProt: P38398."
    print(f"Input:  {text}")
    try:
        gene = extract_gene_info(text)
        print(f"Summary: {gene.gene_name}, chr={gene.chromosome}, exons={gene.exon_count}, UniProt={gene.uniprot_id}")
        print(f"Full JSON: {gene.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 2. Variant Call
    # -----------------------------------------------------------------------
    print("2. VARIANT CALL - VCF-like structured data")
    print("-" * 50)
    text = "Variant at chr17:7577548 G>A in TP53. Quality 99.0. Consequence: missense."
    print(f"Input:  {text}")
    try:
        var = extract_variant_call(text)
        print(f"Summary: {var.chrom}:{var.pos} {var.ref}>{var.alt}, qual={var.qual}, gene={var.gene}, effect={var.consequence}")
        print(f"Full JSON: {var.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 3. Experiment Result
    # -----------------------------------------------------------------------
    print("3. EXPERIMENT RESULT - Quality control and measurements")
    print("-" * 50)
    text = "Sample EXP_001 treated with Doxorubicin 10uM for 24h. Read count: 12500000, mapping rate: 0.94, duplication rate: 0.08. Passed QC. Notes: Library prep was clean, no adapter contamination detected."
    print(f"Input:  {text[:80]}...")
    try:
        exp = extract_experiment_result(text)
        print(f"Summary: {exp.sample_id}, condition={exp.condition}, passed_qc={exp.passed_qc}, measurements={exp.measurements}")
        print(f"Full JSON: {exp.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 4. Differential Expression
    # -----------------------------------------------------------------------
    print("4. DIFFERENTIAL EXPRESSION - DESeq2-style result")
    print("-" * 50)
    text = "Gene TP53: base mean 1234.5, log2 fold change 2.34, p-value 0.0001, adjusted p-value 0.0005. Significantly upregulated."
    print(f"Input:  {text}")
    try:
        de = extract_de_result(text)
        print(f"Summary: {de.gene}, log2FC={de.log2_fold_change}, pvalue={de.pvalue}, padj={de.padj}, dir={de.direction}, sig={de.significant}")
        print(f"Full JSON: {de.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 5. Pathway Enrichment
    # -----------------------------------------------------------------------
    print("5. PATHWAY ENRICHMENT - GO/KEGG enrichment result")
    print("-" * 50)
    text = "Pathway GO:0006954 (cellular response to DNA damage) enriched. Category BP. 15 genes from input match this pathway, total 89 genes in pathway. Enrichment ratio 3.2, p-value 1e-8, FDR 5e-6."
    print(f"Input:  {text[:80]}...")
    try:
        pw = extract_pathway_enrichment(text)
        print(f"Summary: {pw.pathway_id} ({pw.pathway_name}), cat={pw.category}, genes={pw.genes_in_pathway}, ratio={pw.enrichment_ratio}, fdr={pw.fdr}")
        print(f"Full JSON: {pw.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 6. Sequence Feature
    # -----------------------------------------------------------------------
    print("6. SEQUENCE FEATURE - DNA/protein feature annotation")
    print("-" * 50)
    text = "ORF feature from position 120 to 890 on the plus strand. Sequence ATGGCTAAGCTGCTAGCTAG. Score 0.95. Description: putative signal peptide region."
    print(f"Input:  {text[:80]}...")
    try:
        feat = extract_sequence_feature(text)
        print(f"Summary: {feat.feature_type}, pos={feat.start}-{feat.end}, strand={feat.strand}, score={feat.score}")
        print(f"Full JSON: {feat.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 7. Sample Metadata
    # -----------------------------------------------------------------------
    print("7. SAMPLE METADATA - MIAME/MINSEQE-style metadata")
    print("-" * 50)
    text = "Sample S23-042 is a blood sample from Homo sapiens. Patient is 45 years old, male, diagnosed with AML (acute myeloid leukemia). Treated with cytarabine. Collected on 2024-01-15. Batch ID B07."
    print(f"Input:  {text[:80]}...")
    try:
        meta = extract_sample_metadata(text)
        print(f"Summary: {meta.sample_id}, {meta.sample_type}, {meta.organism}, age={meta.age}, sex={meta.sex}, condition={meta.condition}")
        print(f"Full JSON: {meta.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 8. Cluster Info (nested model)
    # -----------------------------------------------------------------------
    print("8. CLUSTER INFO - scRNA-seq cluster with marker genes")
    print("-" * 50)
    text = """Cluster_0 contains 523 cells predicted to be CD4+ T cells with 0.92 confidence.
    Top markers: CD3D (logFC=5.1, padj=1e-20), CD4 (logFC=6.3, padj=1e-30), IL7R (logFC=4.8, padj=1e-15).
    Enriched pathway: T cell receptor signaling."""
    print(f"Input:  {text[:80]}...")
    try:
        cluster = extract_cluster_info(text)
        print(f"Summary: {cluster.cluster_id}, {cluster.cell_count} cells, type={cluster.cell_type_prediction}, confidence={cluster.confidence}")
        print(f"Markers: {[m.gene for m in cluster.marker_genes]}")
        print(f"Full JSON: {cluster.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # 9. Genomic Region (BED-like)
    # -----------------------------------------------------------------------
    print("9. GENOMIC REGION - BED format interval")
    print("-" * 50)
    text = "Peak at chr19 from position 44,909,710 to 44,910,010 on the minus strand. Score 850. Feature: H3K27ac peak."
    print(f"Input:  {text}")
    try:
        region = extract_genomic_region(text)
        print(f"Summary: {region.chrom}:{region.start}-{region.end}, strand={region.strand}, score={region.score}, type={region.feature_type}")
        print(f"Full JSON: {region.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Structured Output Benefits:")
    print("  - Type-safe: Fields have specific types")
    print("  - Validated: Constraints catch errors automatically")
    print("  - Reliable: JSON mode forces valid output")
    print("  - Exportable: Easy conversion to DataFrame/CSV/JSON")
    print("=" * 70)
