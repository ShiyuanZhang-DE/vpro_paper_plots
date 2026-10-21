r"""Science-style six-task hardware SR: Multi-View / Single-view / LAPA init.

Reads results/realworld_multiview_lapa_alltasks.csv; 25 rollouts per cell.
The Mean is pooled over all six tasks, weighted by evaluation rollout count.
Uses the middle-panel conventions of plot_realworld_row.py, extended to three
methods. SciencePlots is required; there is no alternative style.

Usage:
    python plot_realworld_multiview_lapa_alltasks.py

LaTeX (built at the full IEEE text width):
    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/realworld_multiview_lapa_alltasks_science.pdf}
      \caption{Hardware success rates over six tasks, with 25 rollouts per
        method and task. Mean pools the six tasks. Categories: novel object,
        place, movement, and background.}
      \label{fig:realworld_multiview_lapa_alltasks}
    \end{figure*}
"""

from pathlib import Path

import numpy as np
import pandas as pd

import style_science as style
import tasks

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "realworld_multiview_lapa_alltasks.csv"
TASK_ORDER = [f"task{i}" for i in range(1, 7)]
METHODS = ["multiview_sr", "side_only_sr", "lapa_init_sr"]
LABELS = {
    "multiview_sr": "Multi-View",
    "side_only_sr": "Single-view",
    "lapa_init_sr": "LAPA init",
}
# First two match the official science row's middle panel; purple is the
# existing LAPA colour in plot_probe_combo.py. All methods use solid fills.
COLORS = {
    "multiview_sr": style.PALETTE[1],
    "side_only_sr": "#F07800",
    "lapa_init_sr": "#A99BC4",
}
BAR_W = 0.25
MEAN_SLOT_UNITS = 0.80  # Same narrower Mean slot as plot_realworld_row.py.


def load():
    df = pd.read_csv(CSV)
    expected_columns = ["task", *METHODS, "n_rollouts"]
    if list(df.columns) != expected_columns or list(df.task) != TASK_ORDER:
        raise ValueError(f"Expected columns {expected_columns} and tasks {TASK_ORDER}")
    if not (df.n_rollouts == 25).all():
        raise ValueError("Expected 25 rollouts per method/task")
    rates = df[METHODS].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError("Success rates must be finite fractions in [0, 1]")
    if not np.allclose(rates * 25, np.round(rates * 25), rtol=0, atol=1e-9):
        raise ValueError("Success rates must represent integer successes in 25 rollouts")

    # Check the overlapping official inputs without modifying either file.
    official = pd.read_csv(HERE / "results" / "realworld_multiview_vs_side.csv").set_index("task")
    current = df.set_index("task")
    if not np.allclose(current.loc[official.index, METHODS[:2]],
                       official[METHODS[:2]], rtol=0, atol=1e-9):
        raise ValueError("Original tasks disagree with the official view comparison")
    alltasks = pd.read_csv(HERE / "results" / "realworld_alltasks.csv")
    five_demo = alltasks[alltasks.n_demos == 5].set_index("task")
    if not np.allclose(current.loc[TASK_ORDER, "multiview_sr"],
                       five_demo.loc[TASK_ORDER, "video_sr"], rtol=0, atol=1e-9):
        raise ValueError("Multi-View disagrees with the official five-demo video results")

    values = {m: current[m].to_numpy(dtype=float) * 100 for m in METHODS}
    means = {m: float(np.average(values[m], weights=df.n_rollouts)) for m in METHODS}
    return values, means


def make_figure(values, means):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=style.figsize("text", ratio=0.33))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.29, top=0.85)
    n = len(TASK_ORDER)
    x = np.arange(n, dtype=float)
    mean_x = n - 0.5 + MEAN_SLOT_UNITS / 2
    widths = np.append(np.full(n, BAR_W), BAR_W * MEAN_SLOT_UNITS)
    for mi, method in enumerate(METHODS):
        offset = (mi - (len(METHODS) - 1) / 2) * BAR_W
        mx = mean_x + offset * MEAN_SLOT_UNITS
        ax.bar(np.append(x + offset, mx), np.append(values[method], means[method]),
               width=widths, facecolor=COLORS[method],
               edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
        ax.text(mx, means[method] - 2, f"{means[method]:.0f}", ha="center", va="top",
                color=style.INK, fontsize=plt.rcParams["font.size"] - 1,
                fontweight="bold", zorder=5)

    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=style.INK_MUTED, linewidth=0.5, zorder=0)
    ax.axvline(n - 0.5, color=style.INK, linewidth=1.1, zorder=4)
    ax.set_xlim(-0.5, n - 0.5 + MEAN_SLOT_UNITS)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.set_ylabel("Success Rate [%]")
    ax.set_xticks(np.append(x, mean_x))
    ax.set_xticklabels([tasks.wrapped(t, short=True) for t in TASK_ORDER]
                       + [r"$\mathbf{Mean}$"])
    ax.tick_params(axis="x", which="both", length=0)
    # The real science style supplies the boxed axes and inward y ticks.
    handles = [Patch(facecolor=COLORS[m],
                     edgecolor=style.MARKER_EDGE, linewidth=0.5, label=LABELS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, borderaxespad=0, borderpad=0,
               handlelength=1.3, handleheight=0.9, handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.25)
    return fig


def main():
    style.apply_style()
    print(f"Science style: {style.__file__}; stack={style.STYLE_STACK}")
    values, means = load()
    print("Six-task hardware comparison (25 rollouts per method/task)")
    for i, task in enumerate(TASK_ORDER):
        print(f"{task:<6} " + "  ".join(f"{LABELS[m]}={values[m][i]:.0f}%" for m in METHODS))
    print("Mean   " + "  ".join(f"{LABELS[m]}={means[m]:.6f}%" for m in METHODS))
    style.save(make_figure(values, means), "realworld_multiview_lapa_alltasks_science")


if __name__ == "__main__":
    main()
