#!/bin/sh
# Rebuild every figure of theory_condensed.tex. Run from this folder.
set -e
for f in fig_r1_train fig_qr_train fig_capacity fig_qr_seeds fig_random_weights fig_pseudospike; do python3 $f.py > ${f}_log.txt; done
for f in toy_net r1_graph toy_graph M_slices M_horiz; do pdflatex -interaction=nonstopmode $f.tex > /dev/null; done
rm -f *.aux *.log
