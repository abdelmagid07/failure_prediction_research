"""Normalized trajectory schema, raw-trajectory parsing, and batch ingest.

Parsers cover the two agents this project uses: ``parse_mini_swe_traj`` for the
third-party ``mini-swe-agent`` (real SWE-bench runs, the default) and
``parse_swe_traj`` for SWE-agent / the ``stage2/devbugs`` dev harness.
"""
