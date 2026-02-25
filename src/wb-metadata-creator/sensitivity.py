"""
Sensitivity label support for FHIR metadata generation.

Loads the Verily SensitivityLabels CodeSystem and provides:
  - Full code → display mapping for all 29 sensitivity labels
  - Helper to determine if a code is a recognized sensitivity label
  - Helper to build the DS4P inline-sec-label extension
  - Sensitivity vocabulary formatted for LLM prompts

The CodeSystem source of truth is:
  product_mgmnt/Metadata/Metadata JSON for Demo/JSON Metadata/Terminology/CodeSystems/SensitivityLabels.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ── Load CodeSystem ───────────────────────────────────────────────────────────

_CODE_SYSTEM_PATH = (
    Path(__file__).parent.parent.parent
    / "product_mgmnt"
    / "Metadata"
    / "Metadata JSON for Demo"
    / "JSON Metadata"
    / "Terminology"
    / "CodeSystems"
    / "SensitivityLabels.json"
)

_DS4P_EXT_URL = "http://hl7.org/fhir/uv/security-label-ds4p/StructureDefinition/extension-inline-sec-label"
_SENSITIVITY_SYSTEM = "http://fhir.verily.com/CodeSystem/SensitivityLabels"


def _load_code_system() -> dict[str, dict]:
    """
    Load the SensitivityLabels CodeSystem JSON and return a
    dict mapping code → {"display": ..., "definition": ...}.
    """
    labels: dict[str, dict] = {}
    try:
        with open(_CODE_SYSTEM_PATH) as f:
            cs = json.load(f)
        for concept in cs.get("concept", []):
            code = concept.get("code", "")
            if code:
                labels[code] = {
                    "display": concept.get("display", code),
                    "definition": concept.get("definition", ""),
                }
    except FileNotFoundError:
        print(f"⚠ SensitivityLabels CodeSystem not found at {_CODE_SYSTEM_PATH}")
        print("  Using built-in fallback labels.")
    except Exception as e:
        print(f"⚠ Error loading SensitivityLabels CodeSystem: {e}")
        print("  Using built-in fallback labels.")

    # Ensure the core labels always exist even if file is missing
    _FALLBACK = {
        "UID": {"display": "Unique Identifier", "definition": "A unique identifier assigned to a participant."},
        "PHI": {"display": "Protected Health Information", "definition": "Health information as defined by HIPAA."},
        "PII": {"display": "Personally Identifiable Information", "definition": "Information that can identify an individual directly."},
        "PII_BUSINESS": {"display": "Business Identifiable Information", "definition": "Information that identifies a business entity."},
        "FINANCIAL": {"display": "Financial Information", "definition": "Monetary amounts, billing totals, or payment-related data."},
        "FREETEXT": {"display": "Free Text", "definition": "Unstructured text that requires DLP scanning."},
        "P_PNAME": {"display": "Patient Name", "definition": "The full or partial name of a person."},
        "P_DOB": {"display": "Patient Date of Birth", "definition": "A patient's full date of birth."},
        "P_DOD": {"display": "Patient Date of Death", "definition": "A patient's full date of death."},
        "P_BIRTHSEX": {"display": "Patient Birth Sex", "definition": "The biological sex recorded at birth."},
        "P_RACEETHNICITY": {"display": "Patient Race/Ethnicity", "definition": "A person's race or ethnicity."},
        "P_SSN": {"display": "Patient Social Security Number", "definition": "A Social Security Number."},
        "P_MRN": {"display": "Patient Medical Record Number", "definition": "A medical record number."},
        "P_EMAIL": {"display": "Patient Email Address", "definition": "An email address."},
        "P_PHONE": {"display": "Patient Phone Number", "definition": "A telephone number."},
        "P_STREETADDR": {"display": "Patient Street Address", "definition": "A street address."},
        "P_POSTALCODE": {"display": "Patient Postal Code", "definition": "A postal code, such as a ZIP code."},
        "P_DATE": {"display": "Patient Date", "definition": "Any date directly related to an individual."},
    }
    for code, info in _FALLBACK.items():
        if code not in labels:
            labels[code] = info

    return labels


# Module-level cache — loaded once on import
SENSITIVITY_LABELS: dict[str, dict] = _load_code_system()
"""Maps code → {"display": ..., "definition": ...} for all known sensitivity labels."""


# ── Public Helpers ────────────────────────────────────────────────────────────

def is_sensitive(code: str) -> bool:
    """Return True if the code is a recognized sensitivity label (not NONE/empty)."""
    return code.strip().upper() in SENSITIVITY_LABELS


def get_display(code: str) -> str:
    """Return the display name for a sensitivity code, or the code itself as fallback."""
    entry = SENSITIVITY_LABELS.get(code.strip().upper())
    return entry["display"] if entry else code


def build_sec_label_extension(code: str) -> Optional[dict]:
    """
    Build a DS4P inline-sec-label extension dict for the given sensitivity code.
    Returns None if code is empty/NONE/not recognized.
    """
    code = code.strip().upper()
    if not code or code == "NONE" or code not in SENSITIVITY_LABELS:
        return None

    return {
        "url": _DS4P_EXT_URL,
        "valueCoding": {
            "system": _SENSITIVITY_SYSTEM,
            "code": code,
            "display": SENSITIVITY_LABELS[code]["display"],
        },
    }


def sensitivity_comment(code: str) -> str:
    """Return a comment string for the element — e.g. 'SENSITIVE — P_BIRTHSEX (Patient Birth Sex)'."""
    code = code.strip().upper()
    if not code or code == "NONE" or code not in SENSITIVITY_LABELS:
        return "NOT SENSITIVE"
    display = SENSITIVITY_LABELS[code]["display"]
    return f"SENSITIVE — {code} ({display})"


# ── LLM Prompt Vocabulary ─────────────────────────────────────────────────────

def sensitivity_vocabulary_for_prompt() -> str:
    """
    Format the full sensitivity vocabulary for inclusion in LLM prompts.
    Groups codes by category for easier understanding.

    Returns a multi-line string suitable for embedding in a system prompt.
    """
    # Group by category
    hipaa_safe_harbor = []
    demographic = []
    generic = []
    operational = []

    for code, info in SENSITIVITY_LABELS.items():
        line = f"  - `{code}` — {info['display']}: {info['definition']}"
        if code.startswith("P_") and code not in ("P_BIRTHSEX", "P_RACEETHNICITY"):
            hipaa_safe_harbor.append(line)
        elif code in ("P_BIRTHSEX", "P_RACEETHNICITY"):
            demographic.append(line)
        elif code == "FREETEXT":
            operational.append(line)
        else:
            generic.append(line)

    sections = []
    sections.append("### Sensitivity Label Vocabulary (use these EXACT codes)")
    sections.append("")
    sections.append("**HIPAA Safe Harbor Identifiers:**")
    sections.extend(hipaa_safe_harbor)
    sections.append("")
    sections.append("**Sensitive Demographics:**")
    sections.extend(demographic)
    sections.append("")
    sections.append("**Generic Categories:**")
    sections.extend(generic)
    sections.append("")
    sections.append("**Operational:**")
    sections.extend(operational)
    sections.append("")
    sections.append("**Selection Rules:**")
    sections.append("  - Use the MOST SPECIFIC code available (e.g., `P_BIRTHSEX` for sex-at-birth, NOT generic `PHI`)")
    sections.append("  - `P_DOB` for date of birth, `P_DATE` for other person-related dates")
    sections.append("  - `P_MRN` for medical record numbers, `UID` for study-assigned subject IDs")
    sections.append("  - `P_RACEETHNICITY` for race/ethnicity columns")
    sections.append("  - `PHI` only as a fallback for health data that doesn't fit a specific code")
    sections.append("  - `FREETEXT` for any free-text/unstructured text column (triggers DLP scanning)")
    sections.append("  - `NONE` or empty for non-sensitive columns (administrative flags, study metadata)")
    sections.append("  - Absence of a label means NOT SENSITIVE — do not label non-sensitive fields")

    return "\n".join(sections)
