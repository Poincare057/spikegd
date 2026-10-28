# Figures for theory_condensed.tex

Every figure is made by one file here. Python figures share `style.py` (fonts and colours sized for the 5.5 in text width) and
`qif_events.py` (exact event-driven QIF simulation, complex-step event Jacobians, RTRL traces, Benettin/QR). Each script prints the
numbers quoted in its caption; the `*_log.txt` files hold the last run.

| Figure | Source |
|---|---|
| Fig. 1, three neuron network | `toy_net_body.tex` (TikZ; standalone wrapper `toy_net.tex`) |
| Fig. 2(a), R1 current with and without the re-entry | `fig_r1_train.py` -> `r1_train.pdf` |
| Fig. 2(b), R1 computational graph | `r1_graph_body.tex` (`r1_graph.tex`) |
| Fig. 3, paths from A's weights to C's target, blocks of J(k) | `toy_graph_body.tex` (`toy_graph.tex`) |
| App. E, slices of the tensor M | `slices_body.tex`, `horiz_body.tex` (`M_slices.tex`, `M_horiz.tex`) |
| App. F, QR exponents vs. magnitude along a train | `fig_qr_train.py` -> `qr_train.pdf` |
| App. F, capacity (gap count, outputs vs T, spread of c2) | `fig_capacity.py` -> `capacity.pdf` |
| App. F, sources of variation of the QR exponents | `fig_qr_seeds.py` -> `qr_seeds.pdf` |
| App. F, random Gaussian weights (variance g^2/sqrt(l)) | `fig_random_weights.py` -> `random_weights.pdf` |
| Fig. 5, pseudospike probe (uses spikegd; cached in `pseudospike_data.npz`, `--recompute` to rerun) | `fig_pseudospike.py` -> `pseudospike.pdf` |

`./build_figs.sh` rebuilds everything (numpy, scipy, matplotlib, pdflatex; jax and spikegd for the pseudospike figure). Compile the paper from the folder above `figs/`.
