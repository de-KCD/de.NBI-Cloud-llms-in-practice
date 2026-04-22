#!/usr/bin/env python3
"""
Demo 02 Extensions: Interdisciplinary Schemas

This file adds 6 additional schemas for NON-bioinformatics fields:
- Chemistry (ChemicalCompound)
- Clinical/Medical (PatientSample)
- Ecology (SpeciesObservation)
- Neuroscience (ImagingSession)
- Proteomics (MassSpecRun)
- Structural Biology (ProteinStructure)

These schemas follow the same pattern as demo_02_structured.py but are
tailored to different scientific domains.

[TODO]: add the rest of the other schemas when running

Usage:
    from demo_02_extensions import (
        ChemicalCompound, extract_chemical_compound,
        PatientSample, extract_patient_sample,
        # ... etc
    )
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# =============================================================================
# SETUP (same as demo_02)
# =============================================================================

API_KEY = "your-api-key"  # Replace with your API key
API_BASE = "https://denbi-llm-api.bihealth.org/v1"  # Or your endpoint
MODEL = "qwen3.5-fp8"
MAX_TOKENS = 8192

client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY),
    mode=instructor.Mode.JSON
)


# =============================================================================
# SCHEMA 10: CHEMICAL COMPOUND (Chemistry)
# =============================================================================

class ChemicalCompound(BaseModel):
    """
    Chemical compound information.

    Use this schema for:
    - Molecular property extraction from papers
    - Chemical database entries
    - Reaction product documentation
    """

    name: str = Field(description="Compound name")

    smiles: str = Field(
        description="SMILES notation",
        pattern=r"^[A-Za-z0-9@+\-\[\]\(\)\\/=]+$"
    )

    molecular_weight: float = Field(
        description="Molecular weight (g/mol)",
        gt=0,
        le=5000
    )

    logp: float = Field(
        description="Octanol-water partition coefficient",
        ge=-5,
        le=15
    )

    safety_hazard: Optional[str] = Field(
        default=None,
        description="Safety hazard classification"
    )

    cas_number: Optional[str] = Field(
        default=None,
        description="CAS registry number",
        pattern=r"^\d+-\d+-\d+$"
    )


def extract_chemical_compound(text: str) -> ChemicalCompound:
    """Extract chemical compound info from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=ChemicalCompound,
        messages=[
            {"role": "system", "content": "Extract chemical compound data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 11: PATIENT SAMPLE (Clinical/Medical)
# =============================================================================

class PatientSample(BaseModel):
    """
    Clinical patient sample metadata.

    Use this schema for:
    - Biobank sample annotation
    - Clinical trial data extraction
    - Electronic health record structuring
    """

    patient_id: str = Field(
        description="De-identified patient identifier",
        pattern=r"^[A-Z0-9_-]+$"
    )

    age: int = Field(
        description="Age at sample collection (years)",
        ge=0,
        le=120
    )

    sex: str = Field(
        description="Biological sex",
        pattern=r"^(Male|Female|Intersex|Unknown)$"
    )

    diagnosis: str = Field(description="Primary diagnosis")

    tnm_stage: Optional[str] = Field(
        default=None,
        description="TNM cancer staging"
    )

    treatment: Optional[str] = Field(
        default=None,
        description="Treatment regimen"
    )

    sample_type: str = Field(
        description="Sample type",
        pattern=r"^(Blood|Tissue|Urine|Saliva|CSF|Bone marrow)$"
    )

    collection_date: Optional[str] = Field(
        default=None,
        description="Sample collection date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )


def extract_patient_sample(text: str) -> PatientSample:
    """Extract patient sample metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=PatientSample,
        messages=[
            {"role": "system", "content": "Extract patient sample data. Return ONLY valid JSON. HIPAA-compliant de-identification required."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 12: SPECIES OBSERVATION (Ecology)
# =============================================================================

class SpeciesObservation(BaseModel):
    """
    Species observation from field surveys.

    Use this schema for:
    - Biodiversity monitoring
    - Citizen science data (e.g., iNaturalist)
    - Ecological field notes
    """

    species_name: str = Field(
        description="Scientific name (Genus species)",
        pattern=r"^[A-Z][a-z]+ [a-z]+$"
    )

    common_name: Optional[str] = Field(
        default=None,
        description="Common name"
    )

    count: int = Field(
        description="Number of individuals observed",
        ge=1
    )

    habitat: str = Field(description="Habitat description")

    latitude: float = Field(
        description="Latitude (decimal degrees)",
        ge=-90,
        le=90
    )

    longitude: float = Field(
        description="Longitude (decimal degrees)",
        ge=-180,
        le=180
    )

    observer: str = Field(description="Observer name")

    observation_date: Optional[str] = Field(
        default=None,
        description="Observation date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )

    behavior: Optional[str] = Field(
        default=None,
        description="Observed behavior"
    )


def extract_species_observation(text: str) -> SpeciesObservation:
    """Extract species observation from field notes."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=SpeciesObservation,
        messages=[
            {"role": "system", "content": "Extract species observation data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 13: IMAGING SESSION (Neuroscience)
# =============================================================================

class ImagingSession(BaseModel):
    """
    Neuroimaging session metadata (fMRI, EEG, MEG).

    Use this schema for:
    - BIDS dataset annotation
    - Imaging protocol documentation
    - Multi-session experiment tracking
    """

    subject_id: str = Field(
        description="Subject identifier",
        pattern=r"^sub-[0-9]+$"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier",
        pattern=r"^ses-[0-9]+$"
    )

    modality: str = Field(
        description="Imaging modality",
        pattern=r"^(fMRI|sMRI|DTI|EEG|MEG|PET)$"
    )

    scanner: str = Field(description="Scanner manufacturer and model")

    task: str = Field(description="Experimental task")

    runs: int = Field(
        description="Number of runs",
        ge=1
    )

    tr: Optional[float] = Field(
        default=None,
        description="Repetition time (seconds)",
        gt=0
    )

    te: Optional[float] = Field(
        default=None,
        description="Echo time (milliseconds)",
        gt=0
    )

    voxel_size: Optional[str] = Field(
        default=None,
        description="Voxel size (e.g., '2x2x2 mm')"
    )


def extract_imaging_session(text: str) -> ImagingSession:
    """Extract imaging session metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=ImagingSession,
        messages=[
            {"role": "system", "content": "Extract imaging session metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 14: MASS SPEC RUN (Proteomics/Metabolomics)
# =============================================================================

class MassSpecRun(BaseModel):
    """
    Mass spectrometry run metadata.

    Use this schema for:
    - LC-MS/MS experiment tracking
    - Proteomics data annotation
    - Metabolomics workflow documentation
    """

    run_id: str = Field(
        description="Unique run identifier",
        pattern=r"^MS[0-9]+_[0-9]+$"
    )

    instrument: str = Field(description="Mass spectrometer model")

    mode: str = Field(
        description="Acquisition mode",
        pattern=r"^(DDA|DIA|PRM|SRM|Full scan)$"
    )

    polarity: str = Field(
        description="Ionization polarity",
        pattern=r"^(Positive|Negative|Both)$"
    )

    sample_type: str = Field(description="Sample type")

    injection_volume: float = Field(
        description="Injection volume (μL)",
        gt=0
    )

    gradient_length: float = Field(
        description="LC gradient length (minutes)",
        gt=0
    )

    column: Optional[str] = Field(
        default=None,
        description="LC column specification"
    )

    mass_range: Optional[str] = Field(
        default=None,
        description="Mass scan range (e.g., '300-1500 m/z')"
    )


def extract_mass_spec_run(text: str) -> MassSpecRun:
    """Extract mass spec run metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=MassSpecRun,
        messages=[
            {"role": "system", "content": "Extract mass spec run metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# SCHEMA 15: PROTEIN STRUCTURE (Structural Biology)
# =============================================================================

class ProteinStructure(BaseModel):
    """
    Protein structure metadata (PDB/mmCIF).

    Use this schema for:
    - PDB structure annotation
    - Structure validation tracking
    - Structural biology experiment documentation
    """

    pdb_id: str = Field(
        description="PDB identifier",
        pattern=r"^[0-9][A-Za-z0-9]{3}$"
    )

    title: str = Field(description="Structure title")

    method: str = Field(
        description="Structure determination method",
        pattern=r"^(X-ray diffraction|NMR|Cryo-EM|Electron crystallography)$"
    )

    resolution: Optional[float] = Field(
        default=None,
        description="Resolution (Ångström)",
        gt=0
    )

    r_free: Optional[float] = Field(
        default=None,
        description="R-free value",
        ge=0,
        le=1
    )

    r_work: Optional[float] = Field(
        default=None,
        description="R-work value",
        ge=0,
        le=1
    )

    organism: str = Field(description="Source organism")

    chains: int = Field(
        description="Number of chains",
        ge=1
    )

    ligands: Optional[str] = Field(
        default=None,
        description="Bound ligands (comma-separated)"
    )


def extract_protein_structure(text: str) -> ProteinStructure:
    """Extract protein structure metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_model=ProteinStructure,
        messages=[
            {"role": "system", "content": "Extract protein structure metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 02 EXTENSIONS: Interdisciplinary Schemas")
    print("=" * 70)

    # Chemistry example
    print("\n1. CHEMISTRY - Chemical Compound Extraction")
    print("-" * 50)
    chem_text = "Aspirin (acetylsalicylic acid) has molecular weight 180.16 g/mol, logP of 1.19, and CAS number 50-78-2. It's classified as an irritant."
    compound = extract_chemical_compound(chem_text)
    print(f"Compound: {compound.name}")
    print(f"MW: {compound.molecular_weight} g/mol")
    print(f"logP: {compound.logp}")
    print(f"CAS: {compound.cas_number}")
    print(f"Hazard: {compound.safety_hazard}")

    # Clinical example
    print("\n2. CLINICAL - Patient Sample Extraction")
    print("-" * 50)
    patient_text = "Patient PT001, a 65-year-old female, was diagnosed with breast carcinoma (T2N1M0). Blood sample collected on 2024-03-15. Treatment: Doxorubicin + Cyclophosphamide."
    patient = extract_patient_sample(patient_text)
    print(f"Patient: {patient.patient_id}")
    print(f"Age: {patient.age}, Sex: {patient.sex}")
    print(f"Diagnosis: {patient.diagnosis}")
    print(f"Stage: {patient.tnm_stage}")
    print(f"Sample: {patient.sample_type}")

    # Ecology example
    print("\n3. ECOLOGY - Species Observation Extraction")
    print("-" * 50)
    eco_text = "Observed 5 individuals of Quercus robur (English oak) in temperate deciduous forest at coordinates 51.5074, -0.1278. Observer: Dr. Smith. Date: 2024-06-20. Behavior: Flowering."
    obs = extract_species_observation(eco_text)
    print(f"Species: {obs.species_name} ({obs.common_name})")
    print(f"Count: {obs.count}")
    print(f"Habitat: {obs.habitat}")
    print(f"Location: {obs.latitude}, {obs.longitude}")

    print("\n" + "=" * 70)
    print("Try your own examples! See demo_02_structured.py for patterns.")
    print("=" * 70)
