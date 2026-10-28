# Numerics: validation checks

The simulator and all Python figure generators live in `../figs/generate_figs.py`. This folder keeps the two check scripts:

- `checks.py`: out->out Jacobian against eq. (8), the time-to-event identity on all branches, RTRL traces against finite differences of whole runs, the telescoping c2/c_{i+1}.
- `check_wbound.py`: the weight bounds of Prop. 4 for Cases 2 and 3.

Both scripts import the simulator from `../figs/generate_figs.py`; there is no duplicated simulator here. Run `python3 checks.py`
and `python3 check_wbound.py` from this folder.
