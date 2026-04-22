"""
demo_02_structured.py - Structured LLM Outputs for Bioinformatics

This demo shows you how to get STRUCTURED, VALIDATED data from LLMs!

Instead of parsing messy text responses, we use Pydantic models to:
- Define exact schemas for bioinformatics data
- Validate responses automatically (ranges, patterns, types)
- Export to pandas DataFrames, CSV, JSON
- Catch LLM hallucinations before they cause problems

REMEMBER DEMO 01?
We saw that LLMs can:
- Make calculation mistakes
- Return inconsistent formats
- Hallucinate values

THIS DEMO solves those problems with:
- Pydantic schemas (exact structure)
- Validation (catches mistakes)
- Auto-retry (LLM fixes its own errors)

WHY STRUCTURED OUTPUTS MATTER:
Imagine asking an LLM to extract gene info from a paper:

❌ WITHOUT structured output:
   "The BRCA1 gene has 24 exons and is on chromosome 17"
   → You need to parse this text, hope the format is consistent

✅ WITH structured output:
   GeneInfo(gene_name="BRCA1", exon_count=24, chromosome="17")
   → Clean, validated, ready to use in your code!

THIS DEMO COVERS:
- 9 bioinformatics schemas (gene annotation to scRNA-seq clusters)
- Pydantic validation (Field with ge, le, pattern, regex)
- Error handling and retry patterns
- Export to pandas DataFrame and CSV

"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports for environment and JSON handling
# Third-party imports for LLM interaction and data validation

import instructor
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from openai import OpenAI
import json


# =============================================================================
# CONFIGURATION
# =============================================================================
# API credentials and settings for the LLM connection
#
# SECURITY NOTE: In production, use environment variables!
# These are hardcoded here for demo/testing purposes only.

# Your API key
API_KEY = "your-api-key"

# Base URL for the denbi-llm-api service
# API_BASE = "https://denbi-llm-api-internal.bihealth.org/v1"
# or
# API_BASE = "https://denbi-llm-api.bihealth.org/v1"

# Which model to use?
# We will now use the Qwen models that can do structured outputs better
MODEL = "qwen3.5-fp8"

# Max tokens for response
# Increased to account for complex tasks
MAX_TOKENS = 8192

# Timeout for API calls (seconds)
TIMEOUT = 300

# =============================================================================
# LLM CLIENT SETUP
# =============================================================================
# We're using instructor in JSON mode - this is CRITICAL for structured outputs!
#
# What instructor does:
# 1. Converts our Pydantic models to JSON Schema
# 2. Sends schema to LLM as part of the prompt
# 3. LLM generates JSON matching the schema
# 4. instructor parses JSON back into Pydantic objects
# 5. Validates all constraints (ranges, patterns, etc.)
# 6. Retries automatically if validation fails
#
# Why JSON MODE matters:
# - Without it: LLM might return markdown, explanations, or malformed JSON
# - With it: LLM is forced to return ONLY valid JSON matching our schema
# - This is grammar-constrained decoding, not just hoping for valid output
#
# This makes structured outputs RELIABLE - the LLM CAN'T return invalid data!

client = instructor.from_openai(
    OpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        timeout=TIMEOUT
    ),
    mode=instructor.Mode.JSON  # Force JSON output - essential!
)


# =============================================================================
# SCHEMA 1: GENE ANNOTATION (with validation)
# =============================================================================
# This is our first structured output schema!
#
# What is a Pydantic model?
# Think of it as a "data class on steroids" - it defines:
# - What fields the data has
# - What type each field must be
# - Constraints on values (min, max, pattern, etc.)
# - Automatic validation when you create an instance
#
# GeneInfo represents structured gene/protein annotation that you might
# extract from a paper, database, or lab notebook.
# =============================================================================

class GeneInfo(BaseModel):
    """
    Gene/protein annotation from literature or databases.

    This schema captures the essential information about a gene:
    - Official symbol (validated format)
    - Exon count (validated range)
    - Chromosome location
    - Functional description
    - UniProt accession (validated pattern)

    Example usage:
        >>> gene = GeneInfo(gene_name="BRCA1", exon_count=24, chromosome="17")
        >>> print(gene.gene_name)
        BRCA1
    """

    # Gene symbol with STRICT validation
    gene_name: str = Field(
        description="Official gene symbol",
        min_length=1,           # Can't be empty
        max_length=20,          # Reasonable max length
        pattern=r"^[A-Z0-9]+$"  # Must be uppercase letters/numbers only
        # Examples: BRCA1, TP53, EGFR, ACTB
        # Invalid: brca1 (lowercase), BRCA-1 (hyphen), "" (empty)
    )

    # Exon count with numeric range validation
    exon_count: Optional[int] = Field(
        default=None,
        description="Number of exons",
        ge=1,       # Must be >= 1 (genes have at least 1 exon)
        le=10000    # Must be <= 10000 (sanity check)
    )

    # Chromosome - optional, no special validation
    chromosome: Optional[str] = Field(
        default=None,
        description="Chromosome location"
        # Examples: "17", "X", "chr1", "MT"
    )

    # Description - optional text
    description: Optional[str] = Field(
        default=None,
        description="Brief description"
    )

    # UniProt ID with regex pattern validation
    # UniProt IDs follow specific formats:
    # - P12345 (6 chars, starts with P/O/Q)
    # - A0A123BCD (9 chars, mixed)
    uniprot_id: Optional[str] = Field(
        default=None,
        description="UniProt accession",
        pattern=r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
        # Valid: P38398, Q9Y6K9, A0A024R1X9
        # Invalid: ABC123, p38398 (lowercase)
    )


def extract_gene_info(text: str, max_retries: int = 2) -> GeneInfo:
    """
    Extract structured gene information from text with retry on validation error.

    This function asks the LLM to parse unstructured text and return
    a structured GeneInfo object.

    Args:
        text: Unstructured text containing gene information
              Example: "BRCA1 is located on chromosome 17 with 24 exons"
        max_retries: How many times to retry if LLM returns invalid JSON

    Returns:
        GeneInfo: Structured gene annotation

    How it works:
        1. Send text to LLM with instruction to extract gene info
        2. LLM returns JSON matching GeneInfo schema
        3. instructor validates all constraints
        4. If validation fails, retry (LLM might have made a mistake)
        5. Return validated GeneInfo object

    Example:
        >>> text = "The BRCA1 gene on chr17 has 24 exons. UniProt: P38398"
        >>> gene = extract_gene_info(text)
        >>> print(f"{gene.gene_name} has {gene.exon_count} exons")
        BRCA1 has 24 exons
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                response_model=GeneInfo,
                messages=[
                    {"role": "system", "content": "Extract gene information. Return ONLY valid JSON."},
                    {"role": "user", "content": text}
                ]
            )
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed - raise the error
                raise
            # Not the last attempt - try again
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
    # Should never reach here, but just in case
    raise RuntimeError(f"Failed to extract gene info after {max_retries} retries")


# =============================================================================
# SCHEMA 2: EXPERIMENT RESULT (with custom validation)
# =============================================================================
# This schema shows how to add CUSTOM validation logic!
#
# Sometimes Field constraints aren't enough. You might need to:
# - Check relationships between fields
# - Validate complex business logic
# - Ensure data consistency
#
# Pydantic's @field_validator lets you write custom validation functions!
# =============================================================================

class ExperimentResult(BaseModel):
    """
    Experimental measurement results.

    Use this schema to structure lab data like:
    - Sample metadata
    - Treatment conditions
    - Measured values (expression, viability, etc.)
    - Quality control status

    Example:
        >>> exp = ExperimentResult(
        ...     sample_id="S001",
        ...     condition="Drug_A_10uM",
        ...     measurements={"viability": 85.3, "apoptosis": 12.7},
        ...     passed_qc=True
        ... )
    """

    sample_id: str = Field(
        description="Unique sample identifier",
        pattern=r"^[A-Z0-9_-]+$"  # Alphanumeric with underscore/hyphen
        # Valid: S001, Sample-1, CTRL_A
        # Invalid: sample 1 (space), S.001 (dot)
    )

    condition: str = Field(description="Treatment condition")
    # Examples: "Control", "Drug_A_10uM", "Heat_shock_42C"

    # Dictionary of measurements - keys are metric names, values are floats
    measurements: Dict[str, float] = Field(
        description="Key-value measurements"
        # Example: {"viability": 85.3, "apoptosis": 12.7, "expression": 2.5}
    )

    passed_qc: bool = Field(description="Whether sample passed quality control")

    notes: Optional[str] = Field(
        default=None,
        description="Additional observations"
    )

    # CUSTOM VALIDATOR: Ensure all measurement values are positive
    # This runs AFTER basic type validation
    @field_validator('measurements')
    @classmethod
    def validate_measurements(cls, v):
        """
        Ensure all measurement values are positive.

        Args:
            v: The measurements dict

        Returns:
            The same dict if valid

        Raises:
            ValueError: If any measurement is negative

        Why negative values are bad:
        - Viability, expression, concentration can't be negative
        - Negative values indicate measurement errors or data corruption
        """
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Measurement '{key}' cannot be negative: {value}")
        return v


def extract_experiment_result(text: str) -> ExperimentResult:
    """Extract structured experiment result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=ExperimentResult,
        messages=[
            {"role": "system", "content": "Extract experiment results. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 3: VARIANT CALL (VCF-like with validation)
# =============================================================================
# This schema mimics the VCF (Variant Call Format) - the standard for
# reporting genetic variants!
#
# VCF is used by:
# - GATK (Genome Analysis Toolkit)
# - bcftools
# - Clinical variant calling pipelines
#
# Our schema captures the essential VCF fields:
# CHROM, POS, REF, ALT, QUAL, FILTER, INFO
# =============================================================================

class VariantCall(BaseModel):
    """
    Genetic variant/SNP call.

    This represents a single nucleotide variant or small indel,
    similar to a row in a VCF file.

    Example:
        >>> var = VariantCall(
        ...     chrom="17",
        ...     pos=7577548,
        ...     ref="G",
        ...     alt="A",
        ...     qual=99.0,
        ...     gene="TP53",
        ...     consequence="missense"
        ... )
    """

    # Chromosome with flexible pattern (with or without "chr" prefix)
    chrom: str = Field(
        description="Chromosome",
        pattern=r"^(chr)?([0-9]{1,2}|X|Y|M)$"
        # Valid: "17", "chr17", "X", "chrX", "M", "chrM"
        # Invalid: "chromosome17", "1", "MT"
    )

    # Position must be positive (1-based coordinate system)
    pos: int = Field(description="1-based position", ge=1)

    # Reference allele - must be valid DNA
    ref: str = Field(
        description="Reference allele",
        pattern=r"^[ACGTN]+$"
        # Valid: "A", "ATG", "N" (unknown base)
        # Invalid: "U" (RNA), "B" (not a base)
    )

    # Alternate allele - can be "." for no-call
    alt: str = Field(
        description="Alternate allele",
        pattern=r"^[ACGTN.]+$"
        # Valid: "T", "ATG", "." (no-call)
    )

    # Quality score: 0-999 (Phred-scaled)
    qual: Optional[float] = Field(
        default=None,
        description="Quality score",
        ge=0,
        le=999
    )

    filter: Optional[str] = Field(
        default=None,
        description="Filter status (PASS, etc.)"
        # Examples: "PASS", "LowQual", "StrandBias"
    )

    gene: Optional[str] = Field(default=None, description="Affected gene")

    consequence: Optional[str] = Field(
        default=None,
        description="Variant effect"
        # Examples: "missense", "nonsense", "synonymous", "frameshift"
    )

    zygosity: Optional[str] = Field(
        default=None,
        description="Homozygous/heterozygous"
        # Examples: "homozygous", "heterozygous", "hemizygous"
    )


def extract_variant_call(text: str) -> VariantCall:
    """Extract variant call from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=VariantCall,
        messages=[
            {"role": "system", "content": "Extract variant call information. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 4: DIFFERENTIAL EXPRESSION (with statistical validation)
# =============================================================================
# RNA-seq differential expression results!
#
# This schema captures the output of tools like:
# - DESeq2
# - edgeR
# - limma-voom
#
# Key statistical concepts:
# - log2FoldChange: How much expression changed (log2 scale)
# - pvalue: Raw statistical significance
# - padj: P-value adjusted for multiple testing (FDR)
# - significant: Is padj < 0.05?
# =============================================================================

class DifferentialExpression(BaseModel):
    """
    Differential gene expression result.

    Use this for RNA-seq analysis results from DESeq2, edgeR, etc.

    Example:
        >>> de = DifferentialExpression(
        ...     gene="EGFR",
        ...     base_mean=1250.0,
        ...     log2_fold_change=3.5,
        ...     pvalue=0.0001,
        ...     padj=0.005,
        ...     significant=True,
        ...     direction="up"
        ... )
    """

    gene: str = Field(description="Gene symbol")

    # Mean expression across all samples (must be non-negative)
    base_mean: float = Field(
        description="Mean expression across samples",
        ge=0
    )

    # Log2 fold change - can be positive (up) or negative (down)
    log2_fold_change: float = Field(description="Log2 fold change")
    # Examples: 3.5 (8x up), -2.0 (4x down), 0 (no change)

    # P-values must be between 0 and 1
    pvalue: float = Field(description="Raw p-value", ge=0, le=1)

    # Adjusted p-value (FDR) also 0-1
    padj: float = Field(description="Adjusted p-value (FDR)", ge=0, le=1)

    significant: bool = Field(description="Significantly differentially expressed")
    # Usually True if padj < 0.05

    direction: str = Field(description="Up or down regulated")

    # CUSTOM VALIDATOR: Ensure direction matches the sign of log2FC
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        """
        Ensure direction matches log2FC sign.

        This is a cross-field validation - we're checking that the
        stated direction is consistent with the fold change value.

        Why this matters:
        - log2FC > 0 means UP-regulated
        - log2FC < 0 means DOWN-regulated
        - LLMs might get this backwards!
        """
        if v.lower() not in ['up', 'down', 'upregulated', 'downregulated']:
            raise ValueError(f"Invalid direction: {v}. Must be 'up' or 'down'")
        return v.lower()


def extract_de_result(text: str) -> DifferentialExpression:
    """Extract differential expression result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=DifferentialExpression,
        messages=[
            {"role": "system", "content": "Extract differential expression data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 5: PATHWAY ENRICHMENT (GO/KEGG)
# =============================================================================
# Pathway enrichment analysis tells you which biological pathways
# are over-represented in your gene list!
#
# Common tools:
# - g:Profiler
# - DAVID
# - clusterProfiler (R)
# - Enrichr
#
# Pathway IDs follow specific formats:
# - GO:0006915 (Gene Ontology)
# - hsa04151 (KEGG human)
# - mmu04151 (KEGG mouse)
# =============================================================================

class PathwayEnrichment(BaseModel):
    """
    Pathway or GO term enrichment result.

    Use this for pathway analysis results from g:Profiler, DAVID, etc.

    Example:
        >>> pathway = PathwayEnrichment(
        ...     pathway_id="GO:0006915",
        ...     pathway_name="Apoptotic process",
        ...     category="BP",
        ...     genes_in_pathway=["TP53", "BAX", "CASP3"],
        ...     total_genes_in_pathway=150,
        ...     enrichment_ratio=2.5,
        ...     pvalue=0.001,
        ...     fdr=0.01
        ... )
    """

    # Pathway ID with strict pattern validation
    pathway_id: str = Field(
        description="Pathway ID (GO:XXXX or KEGG:XXXX)",
        pattern=r"^(GO:\d{7}|hsa\d{5}|mmu\d{5}|ece\d{5})$"
        # Valid: GO:0006915, hsa04151, mmu04151
        # Invalid: GO:123 (too short), HSA04151 (uppercase)
    )

    pathway_name: str = Field(description="Pathway name")
    # Examples: "Apoptotic process", "PI3K-Akt signaling pathway"

    # Category indicates the type of pathway
    category: str = Field(
        description="Category: BP, MF, CC, or KEGG"
        # BP = Biological Process
        # MF = Molecular Function
        # CC = Cellular Component
        # KEGG = KEGG pathway
    )

    # Which genes from your input list are in this pathway
    genes_in_pathway: List[str] = Field(
        description="Genes from input list in this pathway"
    )

    # Total genes in the pathway (background set)
    total_genes_in_pathway: int = Field(
        description="Total genes in pathway",
        ge=1
    )

    # How enriched the pathway is (ratio of observed/expected)
    enrichment_ratio: float = Field(
        description="Enrichment ratio",
        ge=0
    )

    # Statistical significance
    pvalue: float = Field(description="Enrichment p-value", ge=0, le=1)
    fdr: float = Field(description="False discovery rate", ge=0, le=1)


def extract_pathway_enrichment(text: str) -> PathwayEnrichment:
    """Extract pathway enrichment result from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=PathwayEnrichment,
        messages=[
            {"role": "system", "content": "Extract pathway enrichment data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 6: SEQUENCE FEATURES
# =============================================================================
# This schema annotates features on DNA or protein sequences!
#
# Common feature types:
# - ORFs (Open Reading Frames)
# - Promoters (TATA box, etc.)
# - Transcription factor binding sites
# - Protein domains
# - Restriction enzyme sites
# =============================================================================

class SequenceFeature(BaseModel):
    """
    Annotated feature on a DNA/protein sequence.

    Use this to mark up important regions in your sequence.

    Example:
        >>> feature = SequenceFeature(
        ...     feature_type="TATA_box",
        ...     start=25,
        ...     end=30,
        ...     strand="+",
        ...     sequence="TATAAA",
        ...     score=0.95,
        ...     description="Strong TATA box consensus"
        ... )
    """

    feature_type: str = Field(
        description="Type: ORF, promoter, motif, domain, etc."
    )

    # Start position (1-based, like GenBank format)
    start: int = Field(
        description="Start position (1-based)",
        ge=1
    )

    # End position must be >= start
    end: int = Field(
        description="End position",
        ge=1
    )

    strand: Optional[str] = Field(
        default=None,
        description="Strand: +, -, or N/A"
    )

    sequence: Optional[str] = Field(
        default=None,
        description="Feature sequence"
    )

    # Confidence score 0-1
    score: Optional[float] = Field(
        default=None,
        description="Confidence score",
        ge=0,
        le=1
    )

    description: Optional[str] = Field(
        default=None,
        description="Feature description"
    )

    # CUSTOM VALIDATOR: Ensure end > start
    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """
        Ensure end position is after start.

        This is a cross-field validation - we need to check that
        end > start, which requires accessing both fields.

        Note: In Pydantic v2, use info.data to access other fields.
        """
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError(f"End position ({v}) must be greater than start ({info.data['start']})")
        return v


def extract_sequence_feature(text: str) -> SequenceFeature:
    """Extract sequence feature annotation from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=SequenceFeature,
        messages=[
            {"role": "system", "content": "Extract sequence feature. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 7: SAMPLE METADATA (Clinical/Experimental)
# =============================================================================
# Sample metadata is CRITICAL for reproducible research!
#
# This schema captures:
# - Sample identifiers
# - Biological source (tissue, cell type, organism)
# - Clinical/experimental variables
# - Batch information (for batch effect correction)
#
# Following MIAME/MINSEQE standards for metadata reporting.
# =============================================================================

class SampleMetadata(BaseModel):
    """
    Sample/patient metadata.

    Use this to structure sample information for databases or
    publication supplementary tables.

    Example:
        >>> sample = SampleMetadata(
        ...     sample_id="PT001",
        ...     sample_type="tumor",
        ...     organism="Homo sapiens",
        ...     age=55,
        ...     sex="female",
        ...     condition="breast_cancer",
        ...     treatment="doxorubicin",
        ...     collection_date="2024-01-15",
        ...     batch="B001"
        ... )
    """

    sample_id: str = Field(description="Sample identifier")

    sample_type: str = Field(description="Tissue/cell type")
    # Examples: "tumor", "normal", "blood", "PBMC", "liver"

    organism: str = Field(description="Organism")
    # Examples: "Homo sapiens", "Mus musculus"

    # Age with sanity checks (0-150 years)
    age: Optional[int] = Field(
        default=None,
        description="Age in years",
        ge=0,
        le=150
    )

    sex: Optional[str] = Field(default=None, description="Sex")

    condition: Optional[str] = Field(default=None, description="Disease/condition")

    treatment: Optional[str] = Field(default=None, description="Treatment if any")

    collection_date: Optional[str] = Field(default=None, description="Collection date")

    batch: Optional[str] = Field(
        default=None,
        description="Batch ID"
        # Important for batch effect correction!
    )


def extract_sample_metadata(text: str) -> SampleMetadata:
    """Extract sample metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=SampleMetadata,
        messages=[
            {"role": "system", "content": "Extract sample metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 8: CLUSTERING RESULT (scRNA-seq) - NESTED MODEL!
# =============================================================================
# This shows NESTED Pydantic models - models within models!
#
# Single-cell RNA-seq clustering produces:
# - Clusters of cells (e.g., "Cluster 0", "Cluster 1")
# - Marker genes for each cluster
# - Cell type predictions
#
# We use TWO models:
# 1. ClusterMarker - individual marker gene
# 2. ClusterInfo - the full cluster with list of markers
# =============================================================================

class ClusterMarker(BaseModel):
    """
    Marker gene for a cluster.

    A marker gene is significantly upregulated in one cluster
    compared to all other clusters.

    Example:
        >>> marker = ClusterMarker(gene="CD3D", logfc=4.2, pval_adj=1e-10)
    """

    gene: str = Field(description="Gene symbol")

    # Log fold change - how much higher expression is in this cluster
    logfc: float = Field(description="Log fold change vs other clusters")
    # Typical range: 0.5 (weak) to 10+ (very strong)

    # Adjusted p-value (must be 0-1)
    pval_adj: float = Field(description="Adjusted p-value", ge=0, le=1)


class ClusterInfo(BaseModel):
    """
    Cell cluster from single-cell analysis.

    This represents one cluster from scRNA-seq clustering
    (Seurat, Scanpy, etc.).

    Example:
        >>> cluster = ClusterInfo(
        ...     cluster_id="Cluster_0",
        ...     cell_count=523,
        ...     cell_type_prediction="CD4+ T cell",
        ...     confidence=0.92,
        ...     marker_genes=[
        ...         ClusterMarker(gene="CD3D", logfc=5.1, pval_adj=1e-20),
        ...         ClusterMarker(gene="CD4", logfc=6.3, pval_adj=1e-30),
        ...         ClusterMarker(gene="IL7R", logfc=4.8, pval_adj=1e-15)
        ...     ],
        ...     pathway_enrichment="T cell receptor signaling"
        ... )
    """

    cluster_id: str = Field(description="Cluster identifier")
    # Examples: "Cluster_0", "0", "T_cells"

    cell_count: int = Field(
        description="Number of cells in cluster",
        ge=1  # Can't have empty clusters
    )

    cell_type_prediction: Optional[str] = Field(
        default=None,
        description="Predicted cell type"
    )

    # Confidence in cell type prediction (0-1)
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence 0-1",
        ge=0,
        le=1
    )

    # List of marker genes - THIS IS THE NESTED PART!
    marker_genes: List[ClusterMarker] = Field(
        description="Top marker genes"
    )

    pathway_enrichment: Optional[str] = Field(
        default=None,
        description="Enriched pathways"
    )


def extract_cluster_info(text: str) -> ClusterInfo:
    """Extract cluster information from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=ClusterInfo,
        messages=[
            {"role": "system", "content": "Extract cluster information. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 9: GENOMIC REGION (BED-like)
# =============================================================================
# BED format is the standard for genomic intervals!
#
# Used by:
# - ChIP-seq peak callers (MACS2)
# - ATAC-seq analysis
# - Genome browsers (UCSC, IGV)
# - Region-based operations (bedtools)
#
# BED format (0-based, half-open):
# chrom  start  end  name  score  strand
# chr1   100    200  peak1  50    +
# =============================================================================

class GenomicRegion(BaseModel):
    """
    Genomic coordinate region.

    Similar to BED format - use for peaks, CNV regions, etc.

    Example:
        >>> region = GenomicRegion(
        ...     chrom="chr1",
        ...     start=100000,
        ...     end=100500,
        ...     name="Peak_1",
        ...     score=99.5,
        ...     strand="+",
        ...     feature_type="ChIP-seq peak"
        ... )
    """

    chrom: str = Field(description="Chromosome")

    # Start is 0-based (BED format convention)
    start: int = Field(
        description="Start position (0-based)",
        ge=0
    )

    # End is 1-based, exclusive (BED format)
    end: int = Field(
        description="End position",
        ge=1
    )

    name: Optional[str] = Field(default=None, description="Region name")

    # Score typically 0-1000 in BED format
    score: Optional[float] = Field(
        description="Score 0-1000",
        ge=0,
        le=1000
    )

    strand: Optional[str] = Field(
        default=None,
        description="Strand: +, -, or ."
    )

    feature_type: Optional[str] = Field(
        default=None,
        description="e.g., peak, gene, exon"
    )

    # CUSTOM VALIDATOR: Ensure end > start
    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end position is after start."""
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError(f"End ({v}) must be greater than start ({info.data['start']})")
        return v


def extract_genomic_region(text: str) -> GenomicRegion:
    """Extract genomic region from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=GenomicRegion,
        messages=[
            {"role": "system", "content": "Extract genomic region. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# EXPORT UTILITIES
# =============================================================================
# Once you have structured data, you want to export it!
#
# These helper functions convert your Pydantic models to:
# - pandas DataFrame (for analysis)
# - CSV (for Excel, sharing)
# - JSON (for web APIs, storage)
# =============================================================================

def export_to_dataframe(results: List[dict], schema_name: str):
    """
    Export list of results to pandas DataFrame.

    DataFrames are great for:
    - Statistical analysis
    - Plotting
    - Filtering and sorting
    - Integration with scikit-learn, etc.

    Args:
        results: List of dicts (from model_dump())
        schema_name: Name for display

    Returns:
        pandas DataFrame or None if pandas not available
    """
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        print(f"\n📊 {schema_name} DataFrame:")
        print(df.to_string(index=False))
        return df
    except ImportError:
        print("⚠️  pandas not available - skipping DataFrame export")
        return None


def export_to_csv(results: List[dict], filename: str):
    """
    Export list of results to CSV file.

    CSV is great for:
    - Sharing with collaborators
    - Opening in Excel
    - Long-term storage (simple format)

    Args:
        results: List of dicts
        filename: Output file path
    """
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(results)} records to {filename}")
    except ImportError:
        # Fallback: manual CSV writing (no pandas needed)
        if results:
            with open(filename, 'w') as f:
                headers = list(results[0].keys())
                f.write(','.join(headers) + '\n')
                for row in results:
                    f.write(','.join(str(row.get(h, '')) for h in headers) + '\n')
            print(f"💾 Saved {len(results)} records to {filename} (manual CSV)")


def export_to_json(results: List[dict], filename: str):
    """
    Export list of results to JSON file.

    JSON is great for:
    - Web APIs
    - NoSQL databases
    - Preserving nested structures

    Args:
        results: List of dicts
        filename: Output file path
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved {len(results)} records to {filename}")


# =============================================================================
# SCHEMA SELECTION GUIDE
# =============================================================================
# Quick reference for choosing the right schema!

SCHEMA_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                       STRUCTURED OUTPUT SCHEMA GUIDE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Schema                  │ Use When...                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GeneInfo                │ Extracting gene/protein info from papers, DBs     ║
║  ExperimentResult        │ Parsing lab notebook entries, methods sections    ║
║  VariantCall             │ Extracting SNPs/indels from clinical reports      ║
║  DifferentialExpression  │ Parsing RNA-seq results (DESeq2, edgeR output)    ║
║  PathwayEnrichment       │ GO/KEGG enrichment results from g:Profiler, etc.  ║
║  SequenceFeature         │ ORFs, promoters, motifs, protein domains          ║
║  SampleMetadata          │ Clinical/experimental metadata from publications  ║
║  ClusterInfo             │ scRNA-seq cluster annotations (nested markers!)   ║
║  GenomicRegion           │ ChIP-seq peaks, ATAC-seq peaks, CNV regions       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# DEMO MAIN
# =============================================================================
# Let's see these schemas in action!
#
# This demo:
# 1. Shows the schema guide
# 2. Tests each schema with realistic examples
# 3. Collects results for export demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 02: Structured Outputs with Instructor")
    print("Bioinformatics schemas + validation + export")
    print("=" * 60)
    print()

    # Show schema guide
    print(SCHEMA_GUIDE)
    print()

    # Collect results for export demo
    all_de_results = []
    all_variants = []

    # -------------------------------------------------------------------------
    # Test 1: Gene Info (with validation)
    # -------------------------------------------------------------------------
    print("--- Test 1: Gene Annotation (with validation) ---")
    gene_text = """
    The BRCA1 gene is located on chromosome 17. It contains 24 exons
    spanning approximately 100 kilobases. The protein product is 1863
    amino acids long and is involved in DNA repair. UniProt ID: P38398.
    """
    try:
        result = extract_gene_info(gene_text)
        print(f"✓ Gene: {result.gene_name}, Chr: {result.chromosome}, Exons: {result.exon_count}")
        print(f"  UniProt: {result.uniprot_id}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error (expected if model struggles): {e}")
    print()

    # -------------------------------------------------------------------------
    # Test 2: Experiment Result (with measurement validation)
    # -------------------------------------------------------------------------
    print("--- Test 2: Experiment Result (validated measurements) ---")
    exp_text = """
    Sample S001 was treated with 10uM drug A for 24 hours.
    Cell viability was measured at 85.3% and apoptosis rate at 12.7%.
    The sample passed all quality control checks.
    """
    result = extract_experiment_result(exp_text)
    print(f"✓ Sample: {result.sample_id}, Condition: {result.condition}")
    print(f"  Measurements: {result.measurements}")
    print(f"  QC Passed: {result.passed_qc}")
    print("  Full result:")
    print(f"    {json.dumps(result.model_dump(), indent=2)}")
    print()

    # -------------------------------------------------------------------------
    # Test 3: Variant Call (VCF-like with chromosome validation)
    # -------------------------------------------------------------------------
    print("--- Test 3: Variant Call (VCF-like, validated) ---")
    var_text = """
    A heterozygous missense variant was found in TP53 at chromosome 17
    position 7577548, where the reference G is changed to A.
    Quality score is 99. This is a pathogenic mutation.
    """
    result = extract_variant_call(var_text)
    print(f"✓ Variant: {result.chrom}:{result.pos} {result.ref}>{result.alt}")
    print(f"  Gene: {result.gene}, Qual: {result.qual}, Zygosity: {result.zygosity}")
    print("  Full result:")
    print(f"    {json.dumps(result.model_dump(), indent=2)}")
    all_variants.append(result.model_dump())
    print()

    # -------------------------------------------------------------------------
    # Test 4: Differential Expression (with statistical validation)
    # -------------------------------------------------------------------------
    print("--- Test 4: Differential Expression (validated p-values) ---")
    de_text = """
    Gene EGFR shows significant upregulation with a log2 fold change of 3.5.
    Base mean expression is 1250. The raw p-value is 0.0001 and adjusted
    p-value (FDR) is 0.005.
    """
    result = extract_de_result(de_text)
    print(f"✓ Gene: {result.gene}, Log2FC: {result.log2_fold_change}, Dir: {result.direction}")
    print(f"  P-value: {result.pvalue}, Adjusted: {result.padj}, Significant: {result.significant}")
    print("  Full result:")
    print(f"    {json.dumps(result.model_dump(), indent=2)}")
    all_de_results.append(result.model_dump())
    print()

    # -------------------------------------------------------------------------
    # Test 5: Pathway Enrichment (GO/KEGG)
    # -------------------------------------------------------------------------
    print("--- Test 5: Pathway Enrichment (GO terms) ---")
    pathway_text = """
    GO:0006915 (Apoptotic process) is significantly enriched.
    Category: Biological Process (BP).
    Input genes in pathway: TP53, BAX, CASP3.
    Total genes in pathway: 150.
    Enrichment ratio: 2.5, p-value: 0.001, FDR: 0.01.
    """
    try:
        result = extract_pathway_enrichment(pathway_text)
        print(f"✓ Pathway: {result.pathway_name}")
        print(f"  Genes: {result.genes_in_pathway}")
        print(f"  FDR: {result.fdr}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Test 6: Sequence Feature (ORF/promoter annotation)
    # -------------------------------------------------------------------------
    print("--- Test 6: Sequence Feature Annotation ---")
    feature_text = """
    Found a TATA box promoter element from position 25 to 30 on the + strand.
    Sequence: TATAAA. Confidence score: 0.95.
    This is a strong TATA box consensus.
    """
    try:
        result = extract_sequence_feature(feature_text)
        print(f"✓ Feature: {result.feature_type}")
        print(f"  Position: {result.start}-{result.end}")
        print(f"  Strand: {result.strand}")
        print(f"  Score: {result.score}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Test 7: Sample Metadata (Clinical/Experimental)
    # -------------------------------------------------------------------------
    print("--- Test 7: Sample Metadata ---")
    sample_text = """
    Sample PT001 is a tumor tissue from a 55 year old female patient.
    Organism: Homo sapiens.
    Condition: breast cancer.
    Treatment: doxorubicin.
    Collection date: 2024-01-15.
    Batch: B001.
    """
    try:
        result = extract_sample_metadata(sample_text)
        print(f"✓ Sample: {result.sample_id}")
        print(f"  Type: {result.sample_type}")
        print(f"  Age: {result.age}, Sex: {result.sex}")
        print(f"  Condition: {result.condition}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Test 8: Cluster Info (scRNA-seq with nested markers)
    # -------------------------------------------------------------------------
    print("--- Test 8: scRNA-seq Cluster (nested model) ---")
    cluster_text = """
    Cluster 0 contains 523 cells.
    Predicted cell type: CD4+ T cell with 92% confidence.
    Top marker genes:
    - CD3D: logfc=5.1, pval_adj=1e-20
    - CD4: logfc=6.3, pval_adj=1e-30
    - IL7R: logfc=4.8, pval_adj=1e-15
    Pathway enrichment: T cell receptor signaling.
    """
    try:
        result = extract_cluster_info(cluster_text)
        print(f"✓ Cluster: {result.cluster_id}")
        print(f"  Cells: {result.cell_count}")
        print(f"  Type: {result.cell_type_prediction} ({result.confidence:.0%} confidence)")
        print(f"  Markers: {len(result.marker_genes)} genes")
        for marker in result.marker_genes[:2]:
            print(f"    - {marker.gene}: logfc={marker.logfc}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Test 9: Genomic Region (BED-like coordinates)
    # -------------------------------------------------------------------------
    print("--- Test 9: Genomic Region (BED format) ---")
    region_text = """
    ChIP-seq peak on chromosome 1 from position 100000 to 100500.
    Name: Peak_1. Score: 99.5. Strand: +.
    Feature type: transcription factor binding site.
    """
    try:
        result = extract_genomic_region(region_text)
        print(f"✓ Region: {result.chrom}:{result.start}-{result.end}")
        print(f"  Name: {result.name}")
        print(f"  Score: {result.score}")
        print(f"  Type: {result.feature_type}")
        print("  Full result:")
        print(f"    {json.dumps(result.model_dump(), indent=2)}")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DEMO 02 COMPLETE")
    print("=" * 60)
    print()
    print("All 9 schemas tested!")
    print("  ✓ GeneInfo (with validation)")
    print("  ✓ ExperimentResult (custom validator)")
    print("  ✓ VariantCall (VCF-like)")
    print("  ✓ DifferentialExpression (statistical)")
    print("  ✓ PathwayEnrichment (GO/KEGG)")
    print("  ✓ SequenceFeature (ORF/promoter)")
    print("  ✓ SampleMetadata (clinical)")
    print("  ✓ ClusterInfo (nested scRNA-seq)")
    print("  ✓ GenomicRegion (BED format)")
    print()
    print("Key concepts demonstrated:")
    print("  • Field validation (ge, le, pattern, regex)")
    print("  • Custom validators (@field_validator)")
    print("  • Nested models (ClusterMarker → ClusterInfo)")
    print("  • Optional fields with defaults")
    print("  • Export utilities (DataFrame, CSV, JSON)")
    print()

    # -------------------------------------------------------------------------
    # Export Demo: Show how to save results
    # -------------------------------------------------------------------------
    if all_de_results or all_variants:
        print("=" * 60)
        print("EXPORT DEMO: Saving results to files")
        print("=" * 60)
        print()

        # Export differential expression results
        if all_de_results:
            export_to_dataframe(all_de_results, "Differential Expression")
            export_to_json(all_de_results, "demo02_de_results.json")
            print()

        # Export variant calls
        if all_variants:
            export_to_csv(all_variants, "demo02_variants.csv")
            export_to_json(all_variants, "demo02_variants.json")
            print()

        print("✓ Files created: demo02_de_results.json, demo02_variants.csv, demo02_variants.json")
        print()

    # -------------------------------------------------------------------------
    # Next Steps
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("1. See tutorial for hands-on exercises:")
    print("   → main/tutorial_demo-copy.md (Demo 02 section)")
    print("   → Build your own schemas with validation")
    print()
    print("2. Run demo_03_agent.py")
    print("   → See how agents use tools autonomously")
    print()
    print("3. Combine with demo_01:")
    print("   → Use structured outputs to fix demo_01's reliability issues")
    print()
