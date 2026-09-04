"""Fail the build if a HuggingFace token is committed anywhere.

Notebooks are the risk: a token pasted into a `login(token=...)` cell, or
captured in a cell's stored output, is committed verbatim. Tokens must come
from the environment, a Colab secret, or `hf auth login` -- see
`probe_pipeline.artifacts_io.resolve_hf_token`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# hf_ followed by 20+ base62 chars. Real tokens are 34-40 chars.
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")

# Directories that are not ours to police.
SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", "pts", "artifacts"}

SOURCE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".txt"}


def _walk(suffixes: set[str]):
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        yield path


def test_no_token_literal_in_source():
    offenders = []
    for path in _walk(SOURCE_SUFFIXES):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # This file necessarily contains the pattern itself.
        if path.name == "test_no_secrets.py":
            continue
        if TOKEN_RE.search(text):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"HuggingFace token literal found in: {offenders}"


def test_no_token_in_notebook_source_or_outputs():
    offenders = []
    for path in _walk({".ipynb"}):
        try:
            nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        for i, cell in enumerate(nb.get("cells", [])):
            blob = "".join(cell.get("source", []))
            for out in cell.get("outputs", []) or []:
                blob += "".join(out.get("text", []) or [])
                data = out.get("data", {}) or {}
                for value in data.values():
                    blob += "".join(value) if isinstance(value, list) else str(value)
            if TOKEN_RE.search(blob):
                offenders.append(f"{path.relative_to(REPO)}:cell{i}")
    assert not offenders, f"HuggingFace token found in notebook: {offenders}"


@pytest.mark.parametrize("literal", ["hf_" + "a" * 30, "hf_" + "0Az" * 10])
def test_the_detector_actually_detects(literal: str):
    """Guard against the regex silently rotting into something that matches nothing."""
    assert TOKEN_RE.search(f'login(token="{literal}")')
