"""
Figures from train.py and cosine.py.

  output/figs/fig_loss.svg    A: clean-input loss over training, one panel
                     per rule, one line per sigma (median, interquartile band)
                  B: final loss on held-out jittered inputs vs sigma
  output/figs/fig_cosine.svg  A: per-neuron cosine similarity with the exact
                     gradient (row i of w), over all of training, for every
                     neuron, sigma and approximate rule
                  B: fraction of saved points at which the rule updates row i
  output/figs/fig_cosine_neurons.svg  the same cosine over training for
                     selected neurons (--neurons), one row of panels per rule

Points where a rule leaves row i unchanged have no cosine and are left out.

Usage (from experiments/rnn/local_rules_code/):
    python plot.py
"""

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from output_paths import DATA_DIR, FIGS_DIR, ensure_output_dirs, output_file, svg_path  # noqa: E402

# %%
############################
### Configuration
############################


INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#b5b4ad", "#fcfcfb"
SIGMA_COLOR = {
    0.0: INK,
    0.025: "#86b6ef",
    0.05: "#3987e5",
    0.1: "#1c5cab",
    0.2: "#0d366b",
}
NAMES = {
    "exact": "exact gradient",
    "local": "local gradient",
    "fitted": "local + A + C (incl. off-target)",
}
STYLE = {
    "exact": dict(color=INK, marker="o"),
    "local": dict(color="#2a78d6", marker="s"),
    "fitted": dict(color="#eb6834", marker="D"),
}


# %%
############################
### Plotting
############################


def setup() -> None:
    """
    Sets common plot styling.
    """
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def curves(
    ax: Axes,
    trials: np.ndarray,
    v: np.ndarray,
    sig: np.ndarray,
    ylog: bool,
) -> None:
    """
    Median and interquartile band over seeds (NaN left out), one line per sigma;
    v (S, R, B).
    """
    for si, s in enumerate(sig):
        with warnings.catch_warnings():  # all-NaN slices: no update at that point
            warnings.simplefilter("ignore", RuntimeWarning)
            q1, q3 = np.nanpercentile(v[si], [25, 75], axis=0)
            med = np.nanmedian(v[si], 0)
        if s > 0:
            ax.fill_between(
                trials, q1, q3, color=SIGMA_COLOR[float(s)], alpha=0.2, lw=0
            )
        ax.plot(
            trials,
            med,
            color=SIGMA_COLOR[float(s)],
            lw=1.6 if s == 0 else 1.3,
            label=f"σ = {s:g}",
        )
    ax.set_xscale("symlog", linthresh=trials[1])
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("trial")
    ax.grid(alpha=0.3, lw=0.5, axis="y")


def summary(ax: Axes, sig: np.ndarray, series: list[tuple]) -> None:
    """
    Median and interquartile range vs sigma; series: [(label, values (S, n), style)].
    """
    n = len(series)
    for k, (label, v, st) in enumerate(series):
        med = np.median(v, 1)
        q1, q3 = np.percentile(v, [25, 75], axis=1)
        ax.errorbar(
            np.arange(len(sig)) + (k - (n - 1) / 2) * 0.07,
            med,
            yerr=[med - q1, q3 - med],
            capsize=2.5,
            elinewidth=1,
            lw=1.2,
            ms=5,
            label=label,
            **st,
        )
    ax.set_xticks(np.arange(len(sig)), [f"{s:g}" for s in sig])
    ax.set_xlabel("input jitter σ")
    ax.grid(alpha=0.3, lw=0.5, axis="y")


def fig_loss(args: argparse.Namespace) -> None:
    """
    Plots training loss and final loss on held-out jittered inputs.
    """

    ### Unpack recordings
    d = np.load(args.data)
    sig, trials = d["sigmas"], d["trials"]
    meta = json.loads(str(d["args"]))
    rules = [r for r in NAMES if f"{r}_loss" in d]
    R = d[f"{rules[0]}_loss"].shape[1]
    nc = len(rules)

    ### Training loss
    fig, axes = plt.subplots(
        1, nc + 1, figsize=(3.8 * (nc + 1), 3.6), gridspec_kw={"wspace": 0.3}
    )
    lo = min(np.percentile(d[f"{r}_loss"], 25, axis=1).min() for r in rules)
    hi = max(np.percentile(d[f"{r}_loss"], 75, axis=1).max() for r in rules)
    for c, r in enumerate(rules):
        ax = axes[c]
        curves(ax, trials, d[f"{r}_loss"], sig, ylog=True)
        ax.set_ylim(lo * 0.8, hi * 1.2)
        ax.set_title(f"{'A' if c == 0 else ' '}   {NAMES[r]}", loc="left", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("loss on inputs without jitter")
            ax.legend(frameon=False, fontsize=7, loc="lower left")

    ### Held-out loss
    ax = axes[nc]
    summary(
        ax,
        sig,
        [(NAMES[r], d[f"{r}_heldout"], dict(ls="-", **STYLE[r])) for r in rules],
    )
    ax.set_yscale("log")
    ax.set_ylabel(f"final loss on {meta['n_eval']} held-out jittered inputs")
    ax.set_title("B   final loss vs noise", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(
        "Training under input spike-time jitter "
        f"(AdaBelief, {meta['trials']:,} trials, "
        f"zero init; median and interquartile range over {R} noise seeds, "
        "σ = 0 is a single run)",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=10.5,
    )

    ### Save figure
    args.loss_out = output_file(args.loss_out)
    fig.savefig(args.loss_out, format="svg", bbox_inches="tight", facecolor=BG)
    print(f"  figure -> {args.loss_out}")


def seed_runs(v: np.ndarray, sig: np.ndarray) -> list[np.ndarray]:
    """
    Values (S, R, ...) -> per sigma, the distinct runs only (sigma = 0 is one run).
    """
    return [v[si, :1] if s == 0 else v[si] for si, s in enumerate(sig)]


def fig_cosine(args: argparse.Namespace) -> None:
    """
    Plots the per-neuron cosine similarity over all of training, for every
    neuron, sigma and rule, and how often each row is updated.
    """

    ### Unpack recordings
    d = np.load(args.cosine)
    sig = d["sigmas"]
    ntar = int(d["n_targets"])
    rules = [r for r in ("local", "fitted") if f"{r}_cos" in d]
    N = d[f"{rules[0]}_cos"].shape[-1]
    R = d[f"{rules[0]}_cos"].shape[1]
    x = np.arange(N)

    fig, axes = plt.subplots(
        2,
        len(rules),
        figsize=(5.2 * len(rules), 5.2),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.45, "wspace": 0.2},
    )
    for c, r in enumerate(rules):
        runs = seed_runs(d[f"{r}_cos"], sig)  # per sigma: (R', B, N)
        ax, axf = axes[0, c], axes[1, c]

        ### Cosine per neuron and sigma; fraction of points with an update
        for si, s in enumerate(sig):
            v = runs[si].reshape(-1, N)  # all saved points
            ok = np.isfinite(v)
            upd = ok.mean(0)
            stats = np.full((3, N), np.nan)  # median, q1, q3
            for i in range(N):
                if upd[i] > 0:
                    stats[:, i] = np.percentile(v[ok[:, i], i], [50, 25, 75])
            med, q1, q3 = stats
            off = (si - (len(sig) - 1) / 2) * 0.22
            col = SIGMA_COLOR[float(s)]
            ax.errorbar(
                x + off,
                med,
                yerr=[med - q1, q3 - med],
                color=col,
                marker="o",
                ls="none",
                ms=4.5,
                capsize=2,
                elinewidth=1,
                label=f"σ = {s:g}",
            )
            axf.bar(x + off, upd, width=0.2, color=col)

        ### Axes
        ax.axvspan(-0.5, ntar - 0.5, color=GRID, alpha=0.18, lw=0)
        ax.text(0, 1.12, "targets", color=MUTED, fontsize=7)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_ylim(-1.08, 1.2)
        ax.set_xticks(x)
        ax.set_xlim(-0.6, N - 0.4)
        ax.grid(alpha=0.3, lw=0.5, axis="y")
        ax.set_title(f"{'A' if c == 0 else ' '}   {NAMES[r]}", loc="left", fontsize=9.5)
        axf.set_ylim(0, 1.05)
        axf.set_xticks(x)
        axf.set_xlim(-0.6, N - 0.4)
        axf.set_xlabel("neuron i")
        axf.grid(alpha=0.3, lw=0.5, axis="y")
        axf.set_title(
            f"{'B' if c == 0 else ' '}   "
            "fraction of saved points with an update of row i",
            loc="left",
            fontsize=8.5,
        )
        if c == 0:
            ax.set_ylabel("cosine similarity of row i\nwith exact gradient")
            ax.legend(frameon=False, fontsize=7, loc="lower left")
            axf.set_ylabel("fraction")
    fig.suptitle(
        "Per-neuron cosine similarity of each rule's gradient with the exact gradient "
        "(row i of w; along the rule's own training, inputs without jitter; median "
        f"and interquartile range over saved trials and {R} noise seeds)",
        x=0.01,
        y=1.0,
        ha="left",
        fontsize=10,
    )

    ### Save figure
    args.cosine_out = output_file(args.cosine_out)
    fig.savefig(args.cosine_out, format="svg", bbox_inches="tight", facecolor=BG)
    print(f"  figure -> {args.cosine_out}")


def fig_neurons(args: argparse.Namespace) -> None:
    """
    Plots the per-neuron cosine similarity over training for selected neurons.
    """

    ### Unpack recordings
    d = np.load(args.cosine)
    sig, trials = d["sigmas"], d["trials"]
    ntar = int(d["n_targets"])
    rules = [r for r in ("local", "fitted") if f"{r}_cos" in d]
    nn = len(args.neurons)

    fig, axes = plt.subplots(
        len(rules),
        nn,
        figsize=(3.2 * nn, 2.9 * len(rules)),
        squeeze=False,
        gridspec_kw={"hspace": 0.5, "wspace": 0.25},
    )
    for a, r in enumerate(rules):
        cos = d[f"{r}_cos"]  # (S, R, B, N)
        for b, i in enumerate(args.neurons):
            ax = axes[a, b]
            kind = "target" if i < ntar else "off-target"
            ax.set_title(f"{NAMES[r]}\nneuron {i} ({kind})", loc="left", fontsize=8.5)
            if not np.any(np.isfinite(cos[..., i])):  # never updated by this rule
                ax.text(
                    0.5,
                    0.5,
                    "not updated\nby this rule",
                    ha="center",
                    va="center",
                    color=MUTED,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            curves(ax, trials, cos[..., i], sig, ylog=False)
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color=GRID, lw=0.8)
            if b == 0:
                ax.set_ylabel("cosine similarity of row i\nwith exact gradient")
            if a == 0 and b == 0:
                ax.legend(frameon=False, fontsize=7, loc="lower left")
    R = d[f"{rules[0]}_cos"].shape[1]
    fig.suptitle(
        "Per-neuron cosine similarity with the exact gradient over training (row i "
        "of w: neuron i's input weights; inputs without jitter; median and "
        f"interquartile range over {R} noise seeds; gaps: no update of row i)",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=10,
    )

    ### Save figure
    args.neurons_out = output_file(args.neurons_out)
    fig.savefig(args.neurons_out, format="svg", bbox_inches="tight", facecolor=BG)
    print(f"  figure -> {args.neurons_out}")


# %%
############################
### Command line
############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=str(DATA_DIR / "train_results.npz"))
    parser.add_argument("--cosine", default=str(DATA_DIR / "cosine_results.npz"))
    parser.add_argument(
        "--loss-out", type=svg_path, default=str(FIGS_DIR / "fig_loss.svg")
    )
    parser.add_argument(
        "--cosine-out", type=svg_path, default=str(FIGS_DIR / "fig_cosine.svg")
    )
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[0, 1, 4, 7, 8],
        help="neurons for fig_cosine_neurons.svg (0, 1 are the targets)",
    )
    parser.add_argument(
        "--neurons-out",
        type=svg_path,
        default=str(FIGS_DIR / "fig_cosine_neurons.svg"),
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.loss_out = output_file(args.loss_out)
    args.cosine_out = output_file(args.cosine_out)
    args.neurons_out = output_file(args.neurons_out)
    setup()
    fig_loss(args)
    fig_cosine(args)
    fig_neurons(args)


if __name__ == "__main__":
    main()
