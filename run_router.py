#!/usr/bin/env python3
"""Compatibility entry point: `python run_router.py --config <yaml>`.

Continuity for setups that previously exec'd tools/backend-router/run_router.py.
"""
from handoff_router.cli import main

if __name__ == "__main__":
    main()
