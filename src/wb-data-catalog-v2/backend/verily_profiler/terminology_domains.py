"""
Terminology domain presets and dynamic prompt builder.

Users select domain presets (Clinical, Life Sciences) in the Profiling Wizard.
The selected domains determine which terminology systems are included in the
semantic profiling LLM prompt.
"""

from __future__ import annotations

from typing import Any


TERMINOLOGY_DOMAINS: dict[str, dict[str, Any]] = {
    "clinical": {
        "label": "Clinical",
        "description": "Healthcare, EHR, labs, medications, procedures",
        "systems": {
            "loinc": {
                "uri": "http://loinc.org",
                "use": "lab tests, vital signs, clinical observations",
            },
            "snomed": {
                "uri": "http://snomed.info/sct",
                "use": "clinical findings, procedures, conditions",
            },
            "icd10": {
                "uri": "http://hl7.org/fhir/sid/icd-10",
                "use": "diagnoses and conditions",
            },
            "icd10cm": {
                "uri": "http://hl7.org/fhir/sid/icd-10-cm",
                "use": "US clinical diagnoses (ICD-10-CM)",
            },
            "rxnorm": {
                "uri": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "use": "medications",
            },
            "ndc": {
                "uri": "http://hl7.org/fhir/sid/ndc",
                "use": "drug products",
            },
            "cpt": {
                "uri": "http://www.ama-assn.org/go/cpt",
                "use": "medical procedures",
            },
            "hcpcs": {
                "uri": "https://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets",
                "use": "healthcare procedure codes",
            },
        },
    },
    "life_sciences": {
        "label": "Life Sciences",
        "description": "Drug discovery, proteomics, genomics, bioassays",
        "systems": {
            "chebi": {
                "uri": "https://www.ebi.ac.uk/chebi",
                "use": "chemical compounds, molecular properties, small molecules",
            },
            "pubchem": {
                "uri": "https://pubchem.ncbi.nlm.nih.gov",
                "use": "compound identifiers, bioassay results",
            },
            "uniprot": {
                "uri": "https://www.uniprot.org",
                "use": "protein accessions, gene names, protein function",
            },
            "go": {
                "uri": "http://purl.obolibrary.org/obo/GO",
                "use": "gene ontology terms, biological processes, molecular function",
            },
            "bao": {
                "uri": "https://www.ebi.ac.uk/bioassay-ontology",
                "use": "assay types, endpoints (IC50, Ki, EC50), screening",
            },
            "ncbi_gene": {
                "uri": "http://www.ncbi.nlm.nih.gov/gene",
                "use": "gene identifiers, gene symbols",
            },
        },
    },
}

CUSTOM_SYSTEM = {
    "uri": "urn:verily:custom",
    "use": "organization-specific concepts not in standard terminologies",
}


def get_domains_metadata() -> list[dict[str, Any]]:
    """Return domain metadata for the frontend settings UI."""
    result = []
    for key, domain in TERMINOLOGY_DOMAINS.items():
        systems = [
            {"key": sk, "uri": sv["uri"], "use": sv["use"]}
            for sk, sv in domain["systems"].items()
        ]
        result.append({
            "key": key,
            "label": domain["label"],
            "description": domain["description"],
            "systems": systems,
        })
    return result


def build_terminology_prompt_section(selected_domains: list[str] | None = None) -> str:
    """Build the terminology binding rules for the semantic profiling prompt.

    Only includes systems from the selected domains. If no domains are selected
    or None is passed, falls back to all domains (backward compatibility).
    Custom system is always included.
    """
    if not selected_domains:
        selected_domains = list(TERMINOLOGY_DOMAINS.keys())

    lines = [
        "     TERMINOLOGY BINDING RULES — CRITICAL:",
        "       a) Map the column's CONCEPT (not its raw values) to a standard terminology:",
    ]

    for domain_key in selected_domains:
        domain = TERMINOLOGY_DOMAINS.get(domain_key)
        if not domain:
            continue
        for sys_info in domain["systems"].values():
            lines.append(f"          - {sys_info['uri']} — for {sys_info['use']}")

    lines.extend([
        "       b) If the column concept does NOT map to any standard system, create a",
        "          CUSTOM terminology entry:",
        f"          - system: \"{CUSTOM_SYSTEM['uri']}\"",
        "          - code: a stable snake_case slug describing the concept",
        "            (e.g. \"study_site_identifier\", \"patient_enrollment_status\",",
        "             \"adverse_event_severity_grade\")",
        "          - display: a clear human-readable name for the concept",
        "       c) EVERY column that represents meaningful data MUST get at least one",
        "          binding — either standard or custom.",
        "       d) SKIP terminology bindings ONLY for purely structural columns:",
        "          surrogate keys, auto-increment IDs, system timestamps (created_at,",
        "          updated_at), row version numbers, ETL flags. Set to [] for these.",
        "       e) If an EXISTING REGISTRY is provided below, REUSE entries from it",
        "          when the concept matches. Use the exact same system + code.",
        "       f) Use the ontology system URI but set code to a descriptive slug,",
        "          NOT a specific numeric ID, unless you are certain of the exact code.",
    ])

    return "\n".join(lines)
