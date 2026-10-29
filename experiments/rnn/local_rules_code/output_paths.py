"""Shared output locations anchored to this folder."""

import argparse
from pathlib import Path

# %%
############################
### Output directories
############################


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
FIGS_DIR = OUTPUT_DIR / "figs"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def output_file(path: str | Path) -> str:
    """
    Route bare filenames to data or figures and create the parent directory.

    Paths that include a directory are kept as explicit output overrides.
    """
    path = Path(path)
    if path.parent == Path("."):
        directory = FIGS_DIR if path.suffix.lower() == ".svg" else DATA_DIR
        path = directory / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def svg_path(path: str | Path) -> str:
    """
    Accept only SVG filenames for figure output arguments.
    """
    if Path(path).suffix.lower() != ".svg":
        raise argparse.ArgumentTypeError("figure output must have a .svg extension")
    return str(path)
