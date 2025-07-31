#!/usr/bin/env python3
"""
Main entry point for ArmorCode Integration CLI.

This allows the package to be run as:
    python -m armorcode_integration
    armorcode-integration (console script)
"""

import sys
from .core.main import main


def cli_main():
    """Entry point for console script."""
    import asyncio
    try:
        asyncio.run(main())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main() 