"""
Gemini client for verily-profiler.

Provides model auto-detection with fallback, shared call_gemini(), and JSON extraction.
Tries global endpoint (gemini-3.5-flash) first, falls back to us-central1
(gemini-2.5-flash) if SSL or model-not-found errors occur.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_LOCATION_MODEL_CHAIN = [
    ("us-central1", "gemini-2.5-flash"),
    ("global", "gemini-3.5-flash"),
]

MODEL_LOCATION_MAP = {
    "gemini-2.5-flash": "us-central1",
    "gemini-2.5-pro": "us-central1",
    "gemini-3.5-flash": "global",
    "gemini-3.5-pro": "global",
}

_resolved_location: Optional[str] = None
_resolved_model: Optional[str] = None

_settings_model: Optional[str] = None
_settings_location: Optional[str] = None


def set_model_settings(model: Optional[str], location: Optional[str]):
    """Called by main.py when user saves Settings. Overrides auto-detect."""
    global _settings_model, _settings_location
    _settings_model = model if model else None
    _settings_location = location if location and location != "auto" else None


def _get_client(project_id: Optional[str], location: str):
    from google import genai
    return genai.Client(vertexai=True, project=project_id, location=location)


def detect_available_model(
    project_id: Optional[str] = None,
) -> str:
    """Return the user-configured model, or probe Vertex AI for the best available."""
    if _settings_model:
        return _settings_model
    global _resolved_location, _resolved_model

    if _resolved_model and _resolved_location:
        return _resolved_model

    from google.genai.types import GenerateContentConfig
    config = GenerateContentConfig(temperature=0.1, max_output_tokens=32)

    for location, model in _LOCATION_MODEL_CHAIN:
        try:
            client = _get_client(project_id, location)
            client.models.generate_content(
                model=model,
                contents="What is 2+2? Reply with just the number.",
                config=config,
            )
            _resolved_location = location
            _resolved_model = model
            logger.info(f"Model detected: {model} @ {location}")
            return model
        except Exception as e:
            logger.info(f"Model {model} @ {location} not available: {e}")
            continue

    raise RuntimeError(
        f"No Gemini model available in project {project_id}. "
        f"Tried: {[f'{m}@{l}' for l, m in _LOCATION_MODEL_CHAIN]}. "
        "Check Vertex AI API is enabled and your project has model access."
    )


def _get_location() -> str:
    """Return the resolved location, or default to global."""
    return _resolved_location or "global"


def call_gemini(
    system_prompt: str,
    user_message: str = "",
    model_name: str = "",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 65536,
) -> str:
    """Call Gemini via the google-genai SDK (Vertex AI backend)."""
    from google.genai.types import GenerateContentConfig

    model = model_name or _settings_model or _resolved_model or "gemini-2.5-flash"
    loc = location or _settings_location or MODEL_LOCATION_MAP.get(model) or _get_location()

    client = _get_client(project_id, loc)
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=config,
        )
        return response.text
    except Exception as e:
        if _resolved_location == "global" and ("SSL" in str(e) or "certificate" in str(e)):
            logger.warning(f"SSL error on global endpoint, falling back to us-central1: {e}")
            fallback_client = _get_client(project_id, "us-central1")
            response = fallback_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message,
                config=config,
            )
            return response.text
        raise


def call_gemini_stream(
    system_prompt: str,
    user_message: str = "",
    model_name: str = "",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> Iterator[str]:
    """Yield text chunks from Gemini via streaming."""
    from google.genai.types import GenerateContentConfig

    model = model_name or _settings_model or _resolved_model or "gemini-2.5-flash"
    loc = location or _settings_location or MODEL_LOCATION_MAP.get(model) or _get_location()

    client = _get_client(project_id, loc)
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=user_message, config=config,
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        if _resolved_location == "global" and ("SSL" in str(e) or "certificate" in str(e)):
            logger.warning(f"SSL error on global streaming, falling back to us-central1: {e}")
            fallback_client = _get_client(project_id, "us-central1")
            for chunk in fallback_client.models.generate_content_stream(
                model="gemini-2.5-flash", contents=user_message, config=config,
            ):
                if chunk.text:
                    yield chunk.text
        else:
            raise


def extract_json_from_response(response: str) -> Optional[dict | list]:
    """Extract a JSON object or array from an LLM response."""
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None
