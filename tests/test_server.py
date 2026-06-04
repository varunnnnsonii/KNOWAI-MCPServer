"""Tests for the MCP Server tools and endpoints."""

import pytest

from pydantic import BaseModel
from src.tools.search import _format_search_response


def test_format_search_response_error():
    res = {"error": "Test error", "error_code": "TEST", "suggestion": "Try this"}
    out = _format_search_response(res)
    assert "Error (TEST): Test error" in out
    assert "Suggestion: Try this" in out


def test_format_search_response_empty():
    res = {"results": [], "warning": "No results"}
    out = _format_search_response(res)
    assert out == "No results"


def test_format_search_response_success():
    res = {
        "results": [
            {
                "relevance_score": 0.95,
                "title": "Fix memory leak",
                "category": "Changelog",
                "module": "api",
                "date": "2026-06-01",
                "description": "Fixed an issue with memory",
                "rationale": "App crashed",
                "affected_paths": ["src/api.py", "src/mem.py"],
            }
        ]
    }
    
    out = _format_search_response(res)
    
    assert "--- Result 1 (Score: 0.95) ---" in out
    assert "Title: Fix memory leak" in out
    assert "Category: Changelog | Module: api" in out
    assert "Rationale: App crashed" in out
    assert "Affected paths: src/api.py, src/mem.py" in out


def test_format_search_response_warning():
    res = {
        "results": [{"relevance_score": 0.1, "title": "Low confidence match"}],
        "warning": "Only low confidence matches found."
    }
    
    out = _format_search_response(res)
    assert "WARNING: Only low confidence matches found." in out
    assert "Title: Low confidence match" in out
