r"""Exactly two verified DK1 newtasks MLP motion figures, using Science style.

Old original4 baselines are deliberately excluded: their evaluation data differ.
Read-only consumer of the completed CSV and matching validation manifest.

LaTeX (both figures at full text width; use separate figure* environments):
    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/realworld_motion_r2_o5_A0_A2.pdf}
      \caption{Frozen double-view teacher latents predict realized DK1 newtasks
      end-effector motion over five frames on held-out episodes.}
    \end{figure*}
The ablation uses realworld_motion_r2_o5_A0_A1_A2_A3.pdf at the same width.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import style_science as style

HERE = Path(__file__).resolve().parent
METHODS = {
    'A0': ('A0 — DINO, shared Spatial', '#0072B2', ''),
    'A1': ('A1 — DINO, separate Spatial', '#56B4E9', '///'),
    'A2': ('A2 — LAPA-init, shared Spatial', '#009E73', ''),
    'A3': ('A3 — LAPA-init, separate Spatial', '#D4A017', '///'),
}
GROUPS = [(-1, 'Mean'), (0, r'$\Delta x$'), (1, r'$\Delta y$'),
          (2, r'$\Delta z$'), (3, r'$\Delta\theta_x$'),
          (4, r'$\Delta\theta_y$'), (5, r'$\Delta\theta_z$'), (6, r'$\Delta$grip')]
SPECS = [('realworld_motion_r2_o5_A0_A2', ['A0', 'A2']),
         ('realworld_motion_r2_o5_A0_A1_A2_A3', ['A0', 'A1', 'A2', 'A3'])]


def load(ledger):
    source = Path(ledger['motion_csv'])
    manifest = json.loads(Path(ledger['motion_manifest']).read_text())
    assert manifest['status'] == 'COMPLETE'
    assert manifest['cross_arm_identical_episodes_masks_targets_and_splits'] is True
    assert manifest['target'] == 'state_delta_h5' and manifest['horizon'] == 5
    assert manifest['csv_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    assert set(df.target) == {'state_delta_h5'}
    assert set(df.probe) == {'ridge', 'mlp'} and len(df) == 64
    expected = {(arm, probe, d) for arm in METHODS for probe in ['ridge', 'mlp'] for d, _ in GROUPS}
    assert set(zip(df.method, df.probe, df.action_dim)) == expected
    assert np.isfinite(df.r2).all()
    assert (df.checkpoint_step == 30000).all()
    for (arm, probe), group in df.groupby(['method', 'probe']):
        assert np.isclose(group.loc[group.action_dim == -1, 'r2'].iloc[0], group.loc[group.action_dim >= 0, 'r2'].mean(), rtol=0, atol=1e-12)
        assert group.n_train.nunique() == group.n_test.nunique() == 1
        assert group.n_train.iloc[0] == manifest['arms'][arm]['n_train']
        assert group.n_test.iloc[0] == manifest['arms'][arm]['n_test']
    return df[df.probe == 'mlp'].set_index(['method', 'action_dim']).r2


def figure(scores, arms):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=style.figsize('text', ratio=0.43))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.75)
    x = np.arange(8, dtype=float)
    x[1:] += 0.3
    width = 0.75/len(arms)
    negative = {}
    for i, arm in enumerate(arms):
        label, color, hatch = METHODS[arm]
        values = np.array([scores[arm, d] for d, _ in GROUPS])
        centers = x + (i-(len(arms)-1)/2)*width
        bars = ax.bar(centers, values, width=width, color=color, hatch=hatch,
                      edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3, label=label)
        assert np.allclose([bar.get_height() for bar in bars], values, rtol=0, atol=1e-14)
        for j in np.flatnonzero(values < 0):
            ax.scatter(centers[j], 0.012, marker='v', color=color, edgecolor='black', linewidth=0.4, s=20, zorder=5)
            negative.setdefault(arm, []).append(f'{GROUPS[j][1]}={values[j]:.3f}')
    if negative:
        # Explicit numeric disclosure prevents the requested [0,1] range hiding negatives.
        note = '\n'.join(arm+': '+', '.join(values) for arm, values in negative.items())
        fig.text(0.075, 0.012, 'Below zero (triangles):\n'+note, fontsize=5, va='bottom')
    ax.axvline((x[0]+x[1])/2, linestyle='--', color=style.INK_MUTED, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in GROUPS])
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xlim(-0.65, x[-1]+0.65)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel(r'MLP $R^2$')
    ax.set_xlabel('DK1 newtasks · Front + Side · realized motion over 5 frames', labelpad=8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linewidth=.5)
    handles = [Patch(facecolor=METHODS[a][1], hatch=METHODS[a][2], edgecolor='black', linewidth=.5, label=METHODS[a][0]) for a in arms]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .98), ncol=2, frameon=False, columnspacing=2)
    # Renderer checks catch legend collisions and canvas overflow without tight cropping.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = legend.get_window_extent(renderer)
    assert not lb.overlaps(ax.get_window_extent(renderer)), 'Legend overlaps bars'
    for artist in [legend, ax.xaxis.label, ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels():
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, 'Figure text clipped'
    return fig


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--ledger', type=Path, required=True)
    args = ap.parse_args()
    ledger = json.loads(args.ledger.read_text())
    scores = load(ledger)
    for stem, _ in SPECS:
        for ext in ['pdf', 'png']:
            assert not (style.FIG_DIR/f'{stem}.{ext}').exists(), 'Never overwrite previous figures'
    style.apply_style()
    rendered = [(stem, arms, figure(scores, arms)) for stem, arms in SPECS]
    for stem, arms, fig in rendered:
        print(stem)
        print(scores.loc[arms].to_string())
        style.save(fig, stem)
        for ext in ['pdf', 'png']:
            assert (style.FIG_DIR/f'{stem}.{ext}').stat().st_size > 1000
        import matplotlib.image as mpimg
        pixels = mpimg.imread(style.FIG_DIR/f'{stem}.png')
        assert pixels.shape[0] > 300 and pixels.shape[1] > 1000 and pixels.std() > 0.01
        assert (style.FIG_DIR/f'{stem}.pdf').read_bytes().startswith(b'%PDF')


if __name__ == '__main__':
    main()
