"""
Amazon Bedrock Runtime client.
Pure transport — no LLM reasoning here.
Uses the converse() API with ambient AWS credentials.
"""
import logging
import os
import time
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from app.providers.ollama_client import extract_json  # reuse JSON parser

_log = logging.getLogger("rap.bedrock_client")

DEFAULT_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_MAX_TOKENS = 2048

_RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "RequestTimeout"}


def _region() -> str:
    return os.environ.get(
        "BEDROCK_REGION",
        os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)),
    )


def _model_id() -> str:
    return os.environ.get("BEDROCK_MODEL", DEFAULT_MODEL)


def call_bedrock(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """Call Bedrock converse() API. Returns text content. Retries on throttling."""
    _model = model_id or _model_id()
    _reg = region or _region()
    client = boto3.client("bedrock-runtime", region_name=_reg)

    kwargs: dict = {
        "modelId": _model,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            response = client.converse(**kwargs)
            content = response.get("output", {}).get("message", {}).get("content", [])
            if not content:
                raise ValueError("Empty content in Bedrock response.")
            text = content[0].get("text", "")
            if not text:
                raise ValueError("Empty text in Bedrock response content block.")
            return text
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"]["Message"]
            if code in _RETRYABLE_CODES and attempt < max_retries:
                wait = 2 ** attempt
                _log.warning(f"Bedrock [{code}] attempt {attempt}/{max_retries} — retrying in {wait}s")
                time.sleep(wait)
                last_exc = exc
                continue
            raise RuntimeError(f"Bedrock API error [{code}]: {msg}") from exc
        except BotoCoreError as exc:
            raise RuntimeError(f"Bedrock transport error: {exc}") from exc

    raise RuntimeError(f"Bedrock call failed after {max_retries} retries: {last_exc}") from last_exc


def check_bedrock_available(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """Returns (available, message). Does not raise."""
    _model = model_id or _model_id()
    _reg = region or _region()
    try:
        client = boto3.client("bedrock-runtime", region_name=_reg)
        client.converse(
            modelId=_model,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0.0},
        )
        return True, f"Model '{_model}' available in '{_reg}'."
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        return False, f"Bedrock [{code}]: {msg}"
    except Exception as exc:
        return False, f"Bedrock unavailable ({_reg}): {exc}"
