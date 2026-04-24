"""
demo_02_extensions.py - Interdisciplinary Structured Output Schemas

Continuation of demo_02_structured.py. The same Instructor + pydantic pattern
works for any domain -- the schema is just a structured description of what
information you want extracted. Here we show schemas beyond bioinformatics:
chemistry, clinical, ecology, neuroscience, proteomics, structural biology.

The pattern is identical each time:
  1. Define a pydantic model with Field descriptions (becomes prompt context)
  2. Call client.chat.completions.create with response_model=YourSchema
  3. Get a validated typed object back

Why schemas for different domains: each field has its own conventions and
validations. A clinical sample needs age bounds and HIPAA-aware fields,
a PDB entry needs resolution ranges and R-factor constraints, a mass spec
run needs instrument modes and polarity settings. The schema encodes domain
knowledge so the LLM output is immediately usable, not just plausible text.

"""

import json
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# Setup -- mirrors demo_02_structured.py for consistency.
API_KEY="your-api-key"
API_BASE = "https://denbi-llm-api.bihealth.org/v1"
MODEL = "qwen3.5-fp8"
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

client = instructor.from_openai(
    OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=TIMEOUT),
    mode=instructor.Mode.JSON
)


# ---------------------------------------------------------------------------
# Schema 10: Chemical Compound (Chemistry)
# ---------------------------------------------------------------------------
# Chemistry needs: molecular identifiers (SMILES), physicochemical properties
# (logP, molecular weight), and safety classifications. SMILES is a compact
# text notation for molecular structure -- the regex pattern here is a basic
# sanity check, not a full SMILES validator.

class ChemicalCompound(BaseModel):
    """Chemical compound information extracted from text."""
    name: str = Field(description="Compound name")
    # SMILES: compact text notation for molecular structure.
    # matches: "CC(=O)OC1=CC=CC=C1C(=O)O"  (aspirin)
    # rejects: "aspirin" or "C13H18N2O"     (plain name, formula)
    smiles: str = Field(description="SMILES notation", pattern=r"^[A-Za-z0-9@+\-\[\]\(\\)\\/=]+$")
    # molecular_weight: real compounds are typically 50-5000 g/mol.
    # matches: 180.16  (aspirin)
    # rejects: -10.0 (negative), 99999.0 (unrealistic)
    molecular_weight: float = Field(description="Molecular weight (g/mol)", gt=0, le=5000)
    # logP: octanol-water partition coefficient. Most drugs fall in -5 to 15.
    # matches: 1.19  (aspirin, moderately lipophilic)
    # rejects: 25.0 (outside physiological range)
    logp: float = Field(description="Octanol-water partition coefficient", ge=-5, le=15)
    safety_hazard: Optional[str] = Field(default=None, description="Safety hazard classification")
    # CAS: registry number format "NNN-NN-N" with check digit.
    # matches: "50-78-2"  (aspirin), "68-12-2" (caffeine)
    # rejects: "50782" (no dashes), "ABC-12-3" (letters)
    cas_number: Optional[str] = Field(default=None, description="CAS registry number", pattern=r"^\d+-\d+-\d+$")


def extract_chemical_compound(text: str) -> ChemicalCompound:
    """Extract chemical compound info from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=ChemicalCompound,
        messages=[
            {"role": "system", "content": "Extract chemical compound data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Schema 11: Patient Sample (Clinical/Medical)
# ---------------------------------------------------------------------------
# Clinical data is highly structured by convention. Patient IDs are
# de-identified alphanumerics, ages are bounded, sample types are an enum.
# The TNM staging pattern and sample type enum reflect real clinical
# data standards. Note: in production this would need proper IRB/HIPAA handling.

class PatientSample(BaseModel):
    """Clinical patient sample metadata."""
    # patient_id: de-identified, uppercase alphanumeric.
    # matches: "PT001", "S-42_B", "SAMPLE1"
    # rejects: "pt-001" (lowercase), "Patient 1" (space)
    patient_id: str = Field(description="De-identified patient identifier", pattern=r"^[A-Z0-9_-]+$")
    # age: realistic human range.
    # matches: 65
    # rejects: -1 (negative), 200 (unrealistic)
    age: int = Field(description="Age at sample collection (years)", ge=0, le=120)
    # sex: fixed enum.
    # matches: "Male", "Female"
    # rejects: "male" (lowercase), "M" (abbreviated)
    sex: str = Field(description="Biological sex", pattern=r"^(Male|Female|Intersex|Unknown)$")
    diagnosis: str = Field(description="Primary diagnosis")
    tnm_stage: Optional[str] = Field(default=None, description="TNM cancer staging")
    treatment: Optional[str] = Field(default=None, description="Treatment regimen")
    # sample_type: fixed enum of clinical specimen types.
    # matches: "Blood", "Tissue", "CSF"
    # rejects: "blood" (lowercase), "plasma" (not in enum)
    sample_type: str = Field(description="Sample type", pattern=r"^(Blood|Tissue|Urine|Saliva|CSF|Bone marrow)$")
    # collection_date: ISO date format.
    # matches: "2024-03-15"
    # rejects: "03/15/24" (US format), "2024-3-15" (missing leading zero)
    collection_date: Optional[str] = Field(default=None, description="Sample collection date (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")


def extract_patient_sample(text: str) -> PatientSample:
    """Extract patient sample metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=PatientSample,
        messages=[
            {"role": "system", "content": "Extract patient sample data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Schema 12: Species Observation (Ecology)
# ---------------------------------------------------------------------------
# Ecological field data has a specific structure: binomial nomenclature
# (Genus species), GPS coordinates, counts, and habitat descriptions.
# The latitude/longitude bounds and binomial name pattern enforce data quality.

class SpeciesObservation(BaseModel):
    """Species observation from field surveys."""
    # species_name: binomial nomenclature (Capitalized genus, lowercase species).
    # matches: "Quercus robur", "Homo sapiens"
    # rejects: "quercus robur" (lowercase genus), "English Oak" (common name), "Quercus" (single word)
    species_name: str = Field(description="Scientific name (Genus species)", pattern=r"^[A-Z][a-z]+ [a-z]+$")
    common_name: Optional[str] = Field(default=None, description="Common name")
    # count: number of individuals observed.
    # matches: 5, 100
    # rejects: 0 (nothing observed), -1 (impossible)
    count: int = Field(description="Number of individuals observed", ge=1)
    habitat: str = Field(description="Habitat description")
    # latitude/longitude: decimal degrees within Earth's bounds.
    # matches: 51.5074, -0.1278 (London); -33.8688, 151.2093 (Sydney)
    # rejects: 95.0 (>90 latitude), -200.0 (<-180 longitude)
    latitude: float = Field(description="Latitude (decimal degrees)", ge=-90, le=90)
    longitude: float = Field(description="Longitude (decimal degrees)", ge=-180, le=180)
    observer: str = Field(description="Observer name")
    # observation_date: ISO date format.
    # matches: "2024-06-20"
    # rejects: "June 20 2024" (written), "2024-6-20" (missing leading zero)
    observation_date: Optional[str] = Field(default=None, description="Observation date (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
    behavior: Optional[str] = Field(default=None, description="Observed behavior")


def extract_species_observation(text: str) -> SpeciesObservation:
    """Extract species observation from field notes."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=SpeciesObservation,
        messages=[
            {"role": "system", "content": "Extract species observation data. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Schema 13: Imaging Session (Neuroscience)
# ---------------------------------------------------------------------------
# Neuroimaging metadata follows BIDS (Brain Imaging Data Structure) conventions.
# Subject IDs use the sub-NNN format, modalities are an enum of common
# techniques. TR (repetition time) and TE (echo time) are core acquisition
# parameters for fMRI -- they determine temporal resolution and contrast.

class ImagingSession(BaseModel):
    """Neuroimaging session metadata (fMRI, EEG, MEG)."""
    # subject_id: BIDS convention (sub-NNN).
    # matches: "sub-042", "sub-1", "sub-100"
    # rejects: "S042" (no prefix), "subject-042" (wrong format), "sub-abc" (non-numeric)
    subject_id: str = Field(description="Subject identifier", pattern=r"^sub-[0-9]+$")
    # session_id: BIDS convention (ses-NNN).
    # matches: "ses-01", "ses-2"
    # rejects: "session-1" (wrong format), "ses-abc" (non-numeric)
    session_id: Optional[str] = Field(default=None, description="Session identifier", pattern=r"^ses-[0-9]+$")
    # modality: fixed enum of common neuroimaging techniques.
    # matches: "fMRI", "EEG", "DTI"
    # rejects: "fmri" (lowercase), "MRI" (too generic), "PET scan" (extra word)
    modality: str = Field(description="Imaging modality", pattern=r"^(fMRI|sMRI|DTI|EEG|MEG|PET)$")
    scanner: str = Field(description="Scanner manufacturer and model")
    task: str = Field(description="Experimental task")
    # runs: number of imaging runs in the session.
    # matches: 2 (typical), 10 (long session)
    # rejects: 0 (no data), -1 (impossible)
    runs: int = Field(description="Number of runs", ge=1)
    # tr: repetition time in seconds. Must be positive.
    # matches: 2.0 (standard), 0.5 (fast)
    # rejects: 0 (impossible), -2.0 (negative)
    tr: Optional[float] = Field(default=None, description="Repetition time (seconds)", gt=0)
    # te: echo time in milliseconds. Must be positive.
    # matches: 30.0 (standard), 5.0 (short)
    # rejects: 0 (impossible), -5.0 (negative)
    te: Optional[float] = Field(default=None, description="Echo time (milliseconds)", gt=0)
    voxel_size: Optional[str] = Field(default=None, description="Voxel size (e.g., '2x2x2 mm')")


def extract_imaging_session(text: str) -> ImagingSession:
    """Extract imaging session metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=ImagingSession,
        messages=[
            {"role": "system", "content": "Extract imaging session metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Schema 14: Mass Spec Run (Proteomics/Metabolomics)
# ---------------------------------------------------------------------------
# Mass spectrometry runs are characterized by acquisition mode (DDA = data
# dependent, DIA = data independent), ionization polarity, and LC parameters.
# Run IDs follow lab conventions, instrument models vary by vendor.

class MassSpecRun(BaseModel):
    """Mass spectrometry run metadata."""
    # run_id: lab convention (MS + digits + underscore + digits).
    # matches: "MS001_045", "MS01_1"
    # rejects: "001_045" (no prefix), "MS-001-045" (wrong format), "run_001" (wrong prefix)
    run_id: str = Field(description="Unique run identifier", pattern=r"^MS[0-9]+_[0-9]+$")
    instrument: str = Field(description="Mass spectrometer model")
    # mode: fixed enum of acquisition modes.
    # matches: "DDA", "DIA", "Full scan"
    # rejects: "dda" (lowercase), "Data-dependent" (written out), "PRM mode" (extra word)
    mode: str = Field(description="Acquisition mode", pattern=r"^(DDA|DIA|PRM|SRM|Full scan)$")
    # polarity: ionization polarity.
    # matches: "Positive", "Negative", "Both"
    # rejects: "positive" (lowercase), "+" (abbreviated), "Negative mode" (extra word)
    polarity: str = Field(description="Ionization polarity", pattern=r"^(Positive|Negative|Both)$")
    sample_type: str = Field(description="Sample type")
    # injection_volume: must be positive.
    # matches: 2.0 (typical), 0.5 (nano-flow), 10.0 (large)
    # rejects: 0 (no injection), -2.0 (negative)
    injection_volume: float = Field(description="Injection volume (uL)", gt=0)
    # gradient_length: LC gradient duration in minutes. Must be positive.
    # matches: 60.0 (typical), 5.0 (short), 180.0 (long)
    # rejects: 0 (no gradient), -5.0 (negative)
    gradient_length: float = Field(description="LC gradient length (minutes)", gt=0)
    column: Optional[str] = Field(default=None, description="LC column specification")
    mass_range: Optional[str] = Field(default=None, description="Mass scan range (e.g., '300-1500 m/z')")


def extract_mass_spec_run(text: str) -> MassSpecRun:
    """Extract mass spec run metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=MassSpecRun,
        messages=[
            {"role": "system", "content": "Extract mass spec run metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Schema 15: Protein Structure (Structural Biology)
# ---------------------------------------------------------------------------
# PDB structures have well-defined metadata: the 4-character PDB ID, the
# determination method (X-ray, NMR, cryo-EM), resolution, and R-factors.
# R-free is the key overfitting check -- it should be close to R-work
# and below ~0.3 for a reasonable model.

class ProteinStructure(BaseModel):
    """Protein structure metadata (PDB/mmCIF)."""
    # pdb_id: 4-character PDB code (digit + 3 alphanumeric).
    # matches: "1TIM" (tryptophan synthase), "4HQJ" (human hemoglobin)
    # rejects: "1A" (too short), "ABCDE" (no leading digit), "12345" (too long)
    pdb_id: str = Field(description="PDB identifier", pattern=r"^[0-9][A-Za-z0-9]{3}$")
    title: str = Field(description="Structure title")
    # method: structure determination technique.
    # matches: "X-ray diffraction", "NMR", "Cryo-EM"
    # rejects: "Xray" (missing hyphen), "x-ray" (lowercase), "CryoEM" (missing hyphen)
    method: str = Field(description="Structure determination method", pattern=r"^(X-ray diffraction|NMR|Cryo-EM|Electron crystallography)$")
    # resolution: in Angstroms. Must be positive.
    # matches: 2.0 (typical), 1.5 (high), 10.0 (low, cryo-EM)
    # rejects: 0 (impossible), -2.0 (negative)
    resolution: Optional[float] = Field(default=None, description="Resolution (Angstrom)", gt=0)
    # r_free and r_work: quality metrics [0, 1]. Lower is better.
    # matches: 0.21 (good), 0.30 (acceptable)
    # rejects: 0.0 (impossible), 1.5 (>1, nonsense)
    r_free: Optional[float] = Field(default=None, description="R-free value", ge=0, le=1)
    r_work: Optional[float] = Field(default=None, description="R-work value", ge=0, le=1)
    organism: str = Field(description="Source organism")
    # chains: number of polypeptide chains.
    # matches: 4 (typical tetramer), 1 (monomer), 2 (dimer)
    # rejects: 0 (no chains), -1 (impossible)
    chains: int = Field(description="Number of chains", ge=1)
    ligands: Optional[str] = Field(default=None, description="Bound ligands (comma-separated)")


def extract_protein_structure(text: str) -> ProteinStructure:
    """Extract protein structure metadata from text."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
        response_model=ProteinStructure,
        messages=[
            {"role": "system", "content": "Extract protein structure metadata. Return ONLY valid JSON."},
            {"role": "user", "content": text}
        ]
    )


# ---------------------------------------------------------------------------
# Demo -- test all 6 schemas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 02 EXTENSIONS: Interdisciplinary Structured Output Schemas")
    print("=" * 70)
    print()
    print("All 6 schemas tested below. Each uses the same pattern:")
    print("  pydantic model -> Field descriptions in prompt -> LLM JSON -> validated object")
    print()

    # 1. Chemistry
    print("--- 1. Chemistry: Chemical Compound ---")
    chem_text = "Aspirin (acetylsalicylic acid) has molecular weight 180.16 g/mol, logP of 1.19, and CAS number 50-78-2. It is classified as an irritant."
    compound = extract_chemical_compound(chem_text)
    print(f"  Compound: {compound.name}")
    print(f"  MW: {compound.molecular_weight} g/mol, logP: {compound.logp}")
    print(f"  CAS: {compound.cas_number}, Hazard: {compound.safety_hazard}")
    print(f"  Full JSON: {json.dumps(compound.model_dump(), indent=2)}")
    print()

    # 2. Clinical
    print("--- 2. Clinical: Patient Sample ---")
    patient_text = "Patient PT001, a 65-year-old female, was diagnosed with breast carcinoma (T2N1M0). Blood sample collected on 2024-03-15. Treatment: Doxorubicin + Cyclophosphamide."
    patient = extract_patient_sample(patient_text)
    print(f"  Patient: {patient.patient_id}, Age: {patient.age}, Sex: {patient.sex}")
    print(f"  Diagnosis: {patient.diagnosis}, Stage: {patient.tnm_stage}")
    print(f"  Sample: {patient.sample_type}, Date: {patient.collection_date}")
    print(f"  Full JSON: {json.dumps(patient.model_dump(), indent=2)}")
    print()

    # 3. Ecology
    print("--- 3. Ecology: Species Observation ---")
    eco_text = "Observed 5 individuals of Quercus robur (English oak) in temperate deciduous forest at coordinates 51.5074, -0.1278. Observer: Dr. Smith. Date: 2024-06-20. Behavior: Flowering."
    obs = extract_species_observation(eco_text)
    print(f"  Species: {obs.species_name} ({obs.common_name})")
    print(f"  Count: {obs.count}, Habitat: {obs.habitat}")
    print(f"  Location: {obs.latitude}, {obs.longitude}")
    print(f"  Full JSON: {json.dumps(obs.model_dump(), indent=2)}")
    print()

    # 4. Neuroscience
    print("--- 4. Neuroscience: Imaging Session ---")
    neuro_text = "Subject sub-042 completed a resting-state fMRI session (ses-01) on a Siemens Prisma 3T scanner. 2 runs, TR=2000ms, TE=30ms, voxel size 2x2x2 mm."
    try:
        imaging = extract_imaging_session(neuro_text)
        print(f"  Subject: {imaging.subject_id}, Session: {imaging.session_id}")
        print(f"  Modality: {imaging.modality}, Scanner: {imaging.scanner}")
        print(f"  TR: {imaging.tr}s, TE: {imaging.te}ms, Voxels: {imaging.voxel_size}")
        print(f"  Full JSON: {json.dumps(imaging.model_dump(), indent=2)}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # 5. Proteomics
    print("--- 5. Proteomics: Mass Spec Run ---")
    ms_text = "Mass spec run MS001_045 on Thermo Orbitrap Fusion, DDA mode, positive polarity. Plasma sample, 2 uL injection, 60 min gradient on C18 column, mass range 300-1500 m/z."
    try:
        ms = extract_mass_spec_run(ms_text)
        print(f"  Run: {ms.run_id}, Instrument: {ms.instrument}")
        print(f"  Mode: {ms.mode}, Polarity: {ms.polarity}")
        print(f"  Injection: {ms.injection_volume} uL, Gradient: {ms.gradient_length} min")
        print(f"  Full JSON: {json.dumps(ms.model_dump(), indent=2)}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # 6. Structural Biology
    print("--- 6. Structural Biology: Protein Structure ---")
    struct_text = "PDB entry 1TIM: Tryptophan synthase from Escherichia coli, solved by X-ray diffraction at 2.0 Angstrom resolution. R-free 0.21, R-work 0.18. 4 chains, ligands: NAD, TRY."
    try:
        struct = extract_protein_structure(struct_text)
        print(f"  PDB: {struct.pdb_id}, Method: {struct.method}")
        print(f"  Resolution: {struct.resolution} A, R-free: {struct.r_free}, R-work: {struct.r_work}")
        print(f"  Organism: {struct.organism}, Chains: {struct.chains}, Ligands: {struct.ligands}")
        print(f"  Full JSON: {json.dumps(struct.model_dump(), indent=2)}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    print("=" * 70)
    print("All 6 schemas demonstrated: ChemicalCompound, PatientSample,")
    print("SpeciesObservation, ImagingSession, MassSpecRun, ProteinStructure")
    print("=" * 70)
