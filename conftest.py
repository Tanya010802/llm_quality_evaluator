"""
conftest.py
Shared pytest fixtures available across all test files.
"""

import json
import pytest
from utils.report_builder import ReportBuilder


@pytest.fixture(scope="session")
def test_cases():
    """Load all test cases from the prompts JSON file."""
    with open("prompts/test_cases.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def report():
    """Shared report builder instance — collects results across all tests."""
    builder = ReportBuilder()
    yield builder
    builder.save()  # saves eval_summary.json after all tests complete


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "hallucination: LLM hallucination detection tests")
    config.addinivalue_line("markers", "relevance: Response relevance scoring tests")
    config.addinivalue_line("markers", "format: Output format adherence tests")
    config.addinivalue_line("markers", "consistency: Output consistency across runs tests")
