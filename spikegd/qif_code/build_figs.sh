#!/bin/sh
# Rebuild the six SVG figures.
set -e
cd "$(dirname "$0")"
python3 generate_figs.py "$@"
