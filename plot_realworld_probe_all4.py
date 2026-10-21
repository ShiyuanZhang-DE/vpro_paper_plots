r"""Science-style four-way Fig.7 hardware motion probe.

    z_t -> MLP -> observation.state[t+5] - observation.state[t]

Reads only measured MLP rows: target=state_delta_h5 from the original
results/probe_perdim_realworld.csv, and target=delta_h5 from
results/probe_realworld_lapa_local.csv. No probes or rollouts are recomputed.
Mean is the uniform mean over the seven per-dimension R^2 values, including grip.
SciencePlots is required; there is no alternative style.

Usage:
    python plot_realworld_probe_all4.py

LaTeX (built at the full IEEE column width):
    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/realworld_probe_all4_science.pdf}
      \caption{MLP motion probing on held-out hardware episodes, comparing
        single-view, multi-view, LAPA scratch and LAPA init. Mean averages
        the seven motion dimensions, including gripper state.}
      \label{fig:realworld_probe_all4}
    \end{figure}
"""

from pathlib import Path

import numpy as np
import pandas as pd

import style_science as style

HERE = Path(__file__).resolve().parent
PAPER_CSV = HERE / "results" / "probe_perdim_realworld.csv"
LAPA_CSV = HERE / "results" / "probe_realworld_lapa_local.csv"
METHODS = [
    ("ours_single", "single-view"),
    ("ours_multi", "multi-view"),
    ("lapa_scratch", "LAPA scratch"),
    ("lapa_init", "LAPA init"),
]
GROUPS = [
    (-1, "Mean"), (0, r"$\Delta x$"), (1, r"$\Delta y$"), (2, r"$\Delta z$"),
    (3, r"$\Delta\theta_x$"), (4, r"$\Delta\theta_y$"),
    (5, r"$\Delta\theta_z$"), (6, "grip"),
]
# Preserve the original Fig.7 colours from plot_probe_combo.py; the two
# additional colours are also used there. All methods use solid fills.
COLORS = {
    "ours_single": "#E8833A", "ours_multi": "#C8102E",
    "lapa_scratch": "#7F7F7F", "lapa_init": "#A99BC4",
}
BAR_W, GAP_MEAN = 0.18, 0.35


def load():
    frames = []
    for csv, target, methods in [
        (PAPER_CSV, "state_delta_h5", ["ours_single", "ours_multi"]),
        (LAPA_CSV, "delta_h5", ["lapa_scratch", "lapa_init"]),
    ]:
        df = pd.read_csv(csv)
        rows = df[(df.probe == "mlp") & (df.target == target) & df.method.isin(methods)]
        frames.append(rows[["method", "action_dim", "r2", "n_train", "n_test"]])
    df = pd.concat(frames, ignore_index=True)
    expected = {(method, dim) for method, _ in METHODS for dim, _ in GROUPS}
    actual = list(zip(df.method, df.action_dim))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected exactly one Mean and seven dimension rows per method")
    if not (df.n_train == 137152).all() or not (df.n_test == 35881).all():
        raise ValueError("All four methods must use n_train=137152 and n_test=35881")
    if not np.isfinite(df.r2).all():
        raise ValueError("Probe R^2 values must be finite")
    r2 = df.set_index(["method", "action_dim"]).r2
    for method, _ in METHODS:
        if not np.isclose(r2[(method, -1)], np.mean([r2[(method, i)] for i in range(7)]),
                          rtol=0, atol=1e-6):
            raise ValueError(f"{method}: Mean disagrees with per-dimension R^2")
    print("Probe sample counts:")
    print(df[["method", "n_train", "n_test"]].drop_duplicates().to_string(index=False))
    return r2


def make_figure(r2):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    w = style.COL_WIDTH
    fig, ax = plt.subplots(figsize=style.figsize("col", ratio=0.49))
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.39 / w, right=1 - 0.06 / w,
                        bottom=0.29 / h, top=1 - 0.35 / h)
    x = np.arange(len(GROUPS), dtype=float)
    x[1:] += GAP_MEAN
    for mi, (method, _) in enumerate(METHODS):
        heights = [r2[(method, dim)] for dim, _ in GROUPS]
        ax.bar(x + (mi - (len(METHODS) - 1) / 2) * BAR_W, heights,
               width=BAR_W, facecolor=COLORS[method],
               edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
    ax.axvline((x[0] + x[1]) / 2, color=style.INK_MUTED,
               linewidth=0.6, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in GROUPS])
    # Official realworld_row.py convention: hide major and minor category ticks.
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlim(-0.62, x[-1] + 0.62)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5)
    ax.set_ylabel(r"MLP $R^2$ $\uparrow$", labelpad=1)
    fs = plt.rcParams["font.size"]
    # Typography and legend parameters match plot_realworld_probe_bars.py.
    ax.tick_params(axis="x", labelsize=fs - 1.5, pad=1.5)
    ax.tick_params(axis="y", labelsize=fs - 1.5, pad=1.5)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    # SciencePlots supplies the full box and inward y ticks.
    handles = [Patch(facecolor=COLORS[m],
                     edgecolor=style.MARKER_EDGE, linewidth=0.5, label=label)
               for m, label in METHODS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, fontsize=fs - 1.5, handlelength=0.9,
              handleheight=0.7, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0.1)
    return fig


def main():
    style.apply_style()
    print(f"Science style: {style.__file__}; stack={style.STYLE_STACK}")
    r2 = load()
    print("MLP motion probe (state[t+5] - state[t]):")
    for dim, label in GROUPS:
        print(f"{label:18s} " + "  ".join(f"{m}={r2[(m, dim)]:.6f}" for m, _ in METHODS))
    style.save(make_figure(r2), "realworld_probe_all4_science")


if __name__ == "__main__":
    main()
