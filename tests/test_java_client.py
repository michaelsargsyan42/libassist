"""Runs the Java client (clients/java/LibraryClient.java) against a live API. Skipped if no JDK is installed."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from libassist.config import REPO_ROOT
from libassist.devserver import ApiServer

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(shutil.which("java") is None, reason="no JDK installed")]


def test_java_client_against_the_live_api(settings):
    with ApiServer(settings) as server:
        result = subprocess.run(
            ["java", str(REPO_ROOT / "clients" / "java" / "LibraryClient.java"), server.base_url],
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "1. Top policy passage: loans#renewals" in out
    assert "2. Fines owed: €0.75" in out
    assert "new due date 2026-03-28; retry replayed=true, same body=true" in out
    assert "4. Renewal of L-5002 refused: HTTP 409 renewal-limit" in out
    assert '5. Email updated with If-Match "1", new ETag "2"' in out
    assert "6. Second update with the stale ETag: HTTP 412 precondition-failed" in out
