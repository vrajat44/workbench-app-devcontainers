"""
Prompt Engine for WB Metadata Creator.

Provides:
  1. Shared Gemini client (call_gemini) with fast/pro model helpers
  2. Refinement system prompt for chat-based JSON editing
  3. Utility functions for extracting JSON from LLM responses

Usage:
    from prompt_engine import call_gemini, call_gemini_fast, call_gemini_pro
"""

from __future__ import annotations

import json
import re
from typing import Optional


# ── Model Configuration ──────────────────────────────────────────────────────

FAST_MODEL = "gemini-2.5-flash"       # For generation (fast + cheap)
PRO_MODEL = "gemini-2.5-pro"          # For validation / review (accuracy)
DEFAULT_LOCATION = "us-central1"


# ── Refinement System Prompt ──────────────────────────────────────────────────

REFINEMENT_SYSTEM_PROMPT = """You are a FHIR metadata specialist helping a data steward
refine FHIR StructureDefinition JSON files.

You are given the current state of a FHIR StructureDefinition JSON. The user will ask
you to make specific changes — for example:
- "Change column X to PHI sensitivity"
- "Add a structural link to the diagnoses table"
- "Update the description of column Y"
- "Change the data type of column Z from decimal to integer"

## RULES:
1. Return the COMPLETE modified JSON in a ```json code block
2. Only change what the user asks — preserve everything else exactly
3. Follow the FHIR StructureDefinition format (R4)
4. Security labels use the DS4P extension: http://hl7.org/fhir/uv/security-label-ds4p/StructureDefinition/extension-inline-sec-label
5. Sensitivity codes: UID (Unique Identifier), PHI (Protected Health Information), PII (Personally Identifiable Information)
6. Measurement methods: self-reported, calculated, laboratory-measured, clinician-observed, device-collected, extracted-from-ehr, administrative
7. If the user's request is ambiguous, state your assumptions and proceed
8. Always return valid JSON

## CURRENT JSON:
{current_json}

## EXISTING METADATA CONTEXT (for cross-file references):
{existing_metadata_summary}
"""


# ── Gemini Client ─────────────────────────────────────────────────────────────

def call_gemini(
    system_prompt: str,
    user_message: str,
    model_name: str = FAST_MODEL,
    project_id: Optional[str] = None,
    location: str = DEFAULT_LOCATION,
    temperature: float = 0.1,
    max_output_tokens: int = 65536,
) -> str:
    """
    Call Gemini via the google-genai SDK (Vertex AI backend).

    When running inside a Workbench cloud app, ADC handles auth automatically.
    For local testing, run `gcloud auth application-default login` first.
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )

    return response.text


def call_gemini_fast(
    system_prompt: str,
    user_message: str,
    project_id: Optional[str] = None,
    location: str = DEFAULT_LOCATION,
    temperature: float = 0.1,
    max_output_tokens: int = 65536,
) -> str:
    """Call Gemini with the fast model (generation tasks)."""
    return call_gemini(
        system_prompt=system_prompt,
        user_message=user_message,
        model_name=FAST_MODEL,
        project_id=project_id,
        location=location,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def call_gemini_pro(
    system_prompt: str,
    user_message: str,
    project_id: Optional[str] = None,
    location: str = DEFAULT_LOCATION,
    temperature: float = 0.1,
    max_output_tokens: int = 65536,
) -> str:
    """Call Gemini with the pro model (validation / review tasks)."""
    return call_gemini(
        system_prompt=system_prompt,
        user_message=user_message,
        model_name=PRO_MODEL,
        project_id=project_id,
        location=location,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# ── JSON Extraction ───────────────────────────────────────────────────────────

def extract_json_from_response(response: str) -> Optional[dict]:
    """
    Extract a JSON object from an LLM response that contains a ```json block.
    Returns None if no valid JSON is found.
    """
    # Try ```json ... ``` blocks first
    pattern = r"```json\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try ``` ... ``` blocks without json tag
    pattern = r"```\s*\n(.*?)```"
    for match in re.finditer(pattern, response, re.DOTALL):
        text = match.group(1).strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

    # Try parsing the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None


def extract_multiple_jsons_from_response(response: str) -> list[dict]:
    """
    Extract multiple JSON objects from an LLM response.
    Handles responses with multiple ```json blocks.
    """
    results = []
    pattern = r"```json\s*\n(.*?)```"
    for match in re.finditer(pattern, response, re.DOTALL):
        try:
            obj = json.loads(match.group(1).strip())
            results.append(obj)
        except json.JSONDecodeError:
            continue

    # If no ```json blocks found, try the full response
    if not results:
        single = extract_json_from_response(response)
        if single:
            results.append(single)

    return results


def extract_text_sections_from_response(response: str) -> dict[str, str]:
    """
    Extract named sections from an LLM response.
    Looks for markdown headings (## Section Name) and captures content between them.
    """
    sections = {}
    current_section = "preamble"
    current_lines = []

    for line in response.split("\n"):
        heading_match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if heading_match:
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections
