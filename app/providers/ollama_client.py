"""
Shared Ollama HTTP client.
Handles connection, retries, and JSON extraction.
No LLM reasoning here — pure transport.
"""
import json
import logging
import re
from typing import Optional, Dict, Any

import requests

_log = logging.getLogger("rap.ollama_client")

DEFAULT_TIMEOUT = 180  # seconds — Gemma 3 can be slow on CPU


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.1,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send a generation request to Ollama. Returns raw text response."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            f"Is 'ollama serve' running? Error: {exc}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s. "
            f"Model '{model}' may be too slow or not yet loaded."
        )
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:
            pass
        raise RuntimeError(f"Ollama HTTP error: {exc} — {body}")


def check_ollama_available(base_url: str, model: str) -> tuple[bool, str]:
    """Returns (available, message). Does not raise."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Check exact or prefix match (e.g. "gemma3" matches "gemma3:latest")
        match = any(m == model or m.startswith(model + ":") for m in models)
        if not match:
            return False, (
                f"Model '{model}' not found in Ollama. "
                f"Available: {models}. Run: ollama pull {model}"
            )
        return True, f"Model '{model}' available."
    except Exception as exc:
        return False, f"Ollama unavailable at {base_url}: {exc}"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a string that may contain prose."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # First {...} block
    brace = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    # Greedy last attempt — find outermost {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None
