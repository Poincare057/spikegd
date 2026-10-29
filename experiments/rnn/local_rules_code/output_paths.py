"""Shared output locations, anchored to this folder rather than the working directory."""

import argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
FIGS_DIR = OUTPUT_DIR / "figs"


def ensure_output_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def output_file(path):
    """Create the parent directory for an output, including custom CLI paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def svg_path(path):
    """Accept only SVG filenames for figure output arguments."""
    if Path(path).suffix.lower() != ".svg":
        raise argparse.ArgumentTypeError("figure output must have a .svg extension")
    return str(path)
