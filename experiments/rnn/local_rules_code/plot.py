"""
Figures from train.py and cosine.py.

  output/figs/fig_loss.svg    A: loss on the inputs without jitter over training, one panel
                     per rule, one line per sigma (median, interquartile band)
                  B: final loss on held-out jittered inputs vs sigma
  output/figs/fig_cosine.svg  C: cosine similarity with the exact gradient on the target
                     rows over training, local and fitted rule
                  D: cosine similarity over all of training vs sigma, on the
                     target rows, all rows and off-target rows

Usage (from experiments/rnn/local_rules_code/):

    python plot.py
"""

import argparse
import json

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
SIGMA_COLOR = {0.0: INK, 0.025: "#86b6ef", 0.05: "#3987e5", 0.1: "#1c5cab", 0.2: "#0d366b"}
NAMES = {"exact": "exact gradient", "local": "local gradient",
         "fitted": "local + A + C (incl. off-target)"}
STYLE = {"exact": dict(color=INK, marker="o"), "local": dict(color="#2a78d6", marker="s"),
         "fitted": dict(color="#eb6834", marker="D")}


# %%
############################
### Plotting
############################


def setup() -> None:
    """
    Sets common plot styling.
    """
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED})


def curves(
    ax: Axes,
    trials: np.ndarray,
    v: np.ndarray,
    sig: np.ndarray,
    ylog: bool,
) -> None:
    """
    Median and interquartile band over seeds, one line per sigma; v (S, R, B).
    """
    for si, s in enumerate(sig):
        if s > 0:
            q1, q3 = np.percentile(v[si], [25, 75], axis=0)
            ax.fill_between(trials, q1, q3, color=SIGMA_COLOR[float(s)], alpha=0.2, lw=0)
        ax.plot(trials, np.median(v[si], 0), color=SIGMA_COLOR[float(s)],
                lw=1.6 if s == 0 else 1.3, label=f"σ = {s:g}")
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
        ax.errorbar(np.arange(len(sig)) + (k - (n - 1) / 2) * 0.07, med,
                    yerr=[med - q1, q3 - med], capsize=2.5, elinewidth=1, lw=1.2, ms=5,
                    label=label, **st)
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
    fig, axes = plt.subplots(1, nc + 1, figsize=(3.8 * (nc + 1), 3.6),
                             gridspec_kw={"wspace": 0.3})
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
    summary(ax, sig, [(NAMES[r], d[f"{r}_heldout"], dict(ls="-", **STYLE[r])) for r in rules])
    ax.set_yscale("log")
    ax.set_ylabel(f"final loss on {meta['n_eval']} held-out jittered inputs")
    ax.set_title("B   final loss vs noise", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"Training under input spike-time jitter (AdaBelief, {meta['trials']:,} trials, "
                 f"zero init; median and interquartile range over {R} noise seeds, "
                 "σ = 0 is a single run)", x=0.01, y=1.04, ha="left", fontsize=10.5)
    ### Save figure
    fig.savefig(args.loss_out, format="svg", bbox_inches="tight", facecolor=BG)
    print(f"  figure -> {args.loss_out}")


def fig_cosine(args: argparse.Namespace) -> None:
    """
    Plots cosine similarity along training and across input jitter levels.
    """
    ### Unpack recordings
    d = np.load(args.cosine)
    sig, trials = d["sigmas"], d["trials"]
    rules = [r for r in ("local", "fitted") if r in d]
    R = d[rules[0]].shape[1]
    nc = len(rules)
    ### Similarity along training
    fig, axes = plt.subplots(1, nc + 1, figsize=(4 * (nc + 1), 3.6),
                             gridspec_kw={"width_ratios": [1] * nc + [1.1], "wspace": 0.3})
    for c, r in enumerate(rules):
        ax = axes[c]
        curves(ax, trials, d[r][..., 0], sig, ylog=False)
        ax.set_ylim(-0.4, 1.05)
        ax.axhline(1, color=GRID, lw=0.8)
        ax.set_title(f"{'C' if c == 0 else ' '}   {NAMES[r]}", loc="left", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("cosine similarity with exact gradient\n(target rows)")
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    ### Similarity over all recorded parameters
    series = []
    for r in rules:
        for p, ls in [(0, "-"), (1, ":"), (2, "--")]:
            if r == "local" and p == 2:
                continue  # local does not update off-target rows
            series.append((f"{NAMES[r]}, {str(d['parts'][p])}",
                           d[r][..., p].reshape(len(sig), -1), dict(ls=ls, **STYLE[r])))
    ax = axes[nc]
    summary(ax, sig, series)
    ax.set_ylim(-0.6, 1.05)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_ylabel("cosine similarity with exact gradient")
    ax.set_title("D   over all of training", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Cosine similarity of each rule's update with the exact gradient along the "
                 f"rule's own training (inputs without jitter; median and interquartile range "
                 f"over {R} noise seeds)", x=0.01, y=1.04, ha="left", fontsize=10.5)
    ### Save figure
    fig.savefig(args.cosine_out, format="svg", bbox_inches="tight", facecolor=BG)
    print(f"  figure -> {args.cosine_out}")


# %%
############################
### Command line
############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA_DIR / "train_results.npz"))
    parser.add_argument("--cosine", default=str(DATA_DIR / "cosine_results.npz"))
    parser.add_argument("--loss-out", type=svg_path, default=str(FIGS_DIR / "fig_loss.svg"))
    parser.add_argument("--cosine-out", type=svg_path, default=str(FIGS_DIR / "fig_cosine.svg"))
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.loss_out = output_file(args.loss_out)
    args.cosine_out = output_file(args.cosine_out)
    setup()
    fig_loss(args)
    fig_cosine(args)


if __name__ == "__main__":
    main()
