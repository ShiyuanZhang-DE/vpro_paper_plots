r"""Measured Joint O5 camera-embedding comparison, original SciencePlots renderer.

LaTeX: \begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/joint_o5_camera_embedding_ablation_original4_seed1003.pdf}
\caption{Motion MLP R² of the same teacher with original, zero and swapped camera embeddings.}
\end{figure*}
"""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import plot_realworld_probe_all4 as original

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'repo/code/multiview_spatial_o5'))
from embed_ablation import LEDGER, RUN, STEM, MATCH, read, sha, scores, exclusive

METHODS = [('original', 'Original'), ('zero', 'Zero Camera Embedding'),
           ('swapped', 'Swapped Camera Embedding')]
COLORS = {'original': original.COLORS['ours_multi'],
          'zero': original.COLORS['ours_single'], 'swapped': original.COLORS['lapa_init']}


def load():
    ledger = read(LEDGER)
    baseline = read(ledger['baseline_manifest'])['arms']['ours_multi']
    values = {}
    inputs = {}
    for method, _ in METHODS:
        if method == 'original':
            csv_path, manifest, arm = ledger['baseline_csv'], ledger['baseline_manifest'], 'ours_multi'
        else:
            row = ledger['variants'][method]
            assert read(row['label_audit'])['status'] == 'PASS'
            csv_path, manifest, arm = row['motion_csv'], row['motion_manifest'], method
        checked = scores(csv_path, manifest, arm)
        entry = read(manifest)['arms'][arm]
        for key in MATCH:
            assert entry[key] == baseline[key], key
        if method != 'original':
            assert entry['checkpoint'] == ledger['variants'][method]['checkpoint']
        r2 = checked['mlp']
        values[(method, -1)] = r2['mean']
        values.update({(method, d): v for d, v in enumerate(r2['per_dimension'])})
        inputs[method] = dict(csv=str(csv_path), csv_sha256=sha(csv_path), manifest=str(manifest),
                              manifest_sha256=sha(manifest), mlp_r2=r2)
    return pd.Series(values), inputs


def make_figure(values):
    with patch.multiple(original, METHODS=METHODS, COLORS=COLORS, BAR_W=.75 / 3):
        with patch.object(original.style, 'COL_WIDTH', original.style.TEXT_WIDTH):
            fig = original.make_figure(values)
    ax = fig.axes[0]
    heights = [values[(m, d)] for m, _ in METHODS for d, _ in original.GROUPS]
    low = min(0, np.floor((min(heights) - .025) * 10) / 10) if min(heights) < 0 else 0
    ax.set_ylim(low, max(1, max(heights) + .025))
    ax.set_yticks(([low] if low < 0 else []) + [0, .5, 1])
    if low < 0:
        ax.axhline(0, color=original.style.INK_MUTED, linewidth=.5)
    assert len(ax.patches) == 24
    assert np.allclose([p.get_height() for p in ax.patches], heights, rtol=0, atol=1e-14)
    assert len(ax.get_xticklabels()) == 8
    assert [x.get_text() for x in ax.get_legend().get_texts()] == [label for _, label in METHODS]
    assert ax.get_ylim()[0] <= min(heights)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [ax.get_legend(), ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, 'Clipped text'
    assert not ax.get_legend().get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
    return fig


def main():
    import matplotlib.image as mpimg
    pdf, png = [original.style.FIG_DIR / (STEM + '.' + ext) for ext in ['pdf', 'png']]
    assert not pdf.exists() and not png.exists(), 'Never overwrite a figure'
    assert not (RUN / 'figure_validation.json').exists()
    values, inputs = load()
    original.style.apply_style()
    fig = make_figure(values)
    print('Motion MLP R² (uniform seven-dimension mean):\n' + values.to_string(), flush=True)
    original.style.save(fig, STEM)
    pixels = mpimg.imread(png)
    assert pixels.shape[1] > 1500 and pixels.std() > .01
    assert pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size > 1000
    exclusive(RUN / 'figure_validation.json', dict(status='PASS', methods=METHODS,
        n_groups=8, n_bars=24, probe='mlp', target='state_delta_h5', inputs=inputs,
        artifacts={str(p): sha(p) for p in [pdf, png]},
        visual_inspection='PENDING; inspect actual PNG and rendered PDF after job completion'))


if __name__ == '__main__':
    main()
