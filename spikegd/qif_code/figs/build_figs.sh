#!/bin/sh
# Rebuild every figure of theory_condensed.tex. Run from this folder.
set -e
python3 generate_figs.py --log-dir .
for f in toy_net r1_graph toy_graph M_slices M_horiz; do pdflatex -interaction=nonstopmode $f.tex > /dev/null; done
rm -f *.aux *.log
