# Numerics: validation checks

The simulation and the figure scripts now live in `../figs/` (`qif_events.py` and `fig_*.py`). This folder keeps the two check scripts:

- `checks.py`: out->out Jacobian against eq. (8), the time-to-event identity on all branches, RTRL traces against finite differences of whole runs, the telescoping c2/c_{i+1}.
- `check_wbound.py`: the weight bounds of Prop. 4 for Cases 2 and 3.

`qif_events.py` here is a copy of `../figs/qif_events.py` so the checks run on their own: `python3 checks.py; python3 check_wbound.py`.
