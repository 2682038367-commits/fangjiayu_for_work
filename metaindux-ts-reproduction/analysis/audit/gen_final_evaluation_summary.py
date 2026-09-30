#!/usr/bin/env python3
"""Compatibility entry point for the canonical complete report generator.

The former w48-only implementation is preserved as
``gen_final_evaluation_summary_w48_legacy.py`` for historical inspection.
"""
from gen_complete_reproduction_report import main


if __name__ == "__main__":
    raise SystemExit(main())
