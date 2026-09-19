"""Compatibility shim. Prefer python -m cva.cli drift."""
from cva.detectors.drift.command import main, run  # noqa: F401

if __name__ == '__main__':
    main()
