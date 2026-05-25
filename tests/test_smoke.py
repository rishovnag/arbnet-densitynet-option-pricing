"""End-to-end smoke test for arbnet.

This delegates to scripts/smoke_test.py so the test suite and the script stay
in sync. Slow (~10s); skipped under --fast.
"""
import os
import subprocess
import sys


def test_smoke():
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "..", "scripts", "smoke_test.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, "smoke test failed"
    assert "ALL SMOKE TESTS PASSED" in result.stdout


if __name__ == "__main__":
    test_smoke()
    print("test_smoke: OK")
