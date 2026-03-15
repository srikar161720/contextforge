"""
tests/conftest.py — Shared pytest fixtures and configuration.

Integration tests that make real Bedrock API calls are marked with
@pytest.mark.integration. They are automatically skipped when AWS
credentials are not configured in the environment.
"""

import os

import pytest
from dotenv import load_dotenv

# Load .env from project root so AWS credentials are available for integration tests
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that make real AWS Bedrock API calls "
        "(skipped when credentials are absent)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are long-running (>60s); "
        "run with: pytest -m 'integration and slow'",
    )


@pytest.fixture(scope="session", autouse=True)
def skip_integration_without_credentials(request):
    """Session-scoped fixture: skip integration tests if AWS creds are missing."""
    # This fixture runs once; individual integration tests use the marker check below.


def pytest_runtest_setup(item):
    """Skip any test marked @pytest.mark.integration if AWS creds are absent."""
    if item.get_closest_marker("integration"):
        key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if not key or not secret:
            pytest.skip(
                "Skipping integration test — AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env."
            )


@pytest.fixture(scope="session")
def bedrock_client():
    """Return a BedrockClient instance for integration tests."""
    from infra.bedrock_client import BedrockClient
    return BedrockClient()
