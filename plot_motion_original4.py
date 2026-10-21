r"""Original four historical methods plus original4 o5 A0/A2 or A0-A3.

Reuses plot_realworld_probe_all4.load/make_figure and the original Science style.
Historical CSVs remain unchanged. Sample-range differences and the historical
raw-frame audit limits are documented in repo/code/multiview_spatial_o5/expe_docu.md.

LaTeX: both outputs span the full text width: figure*, width=\textwidth.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import plot_realworld_probe_all4 as original

HERE=Path(__file__).resolve().parent
NEW_METHODS=[('A0','A0 — DINO shared Spatial'),('A1','A1 — DINO separate Spatial'),
             ('A2','A2 — LAPA-init shared Spatial'),('A3','A3 — LAPA-init separate Spatial')]
COLORS=dict(original.COLORS,A0='#0072B2',A1='#56B4E9',A2='#009E73',A3='#D4A017')
SPECS=[('realworld_motion_r2_o5_original_4_A0_A2',['A0','A2']),
       ('realworld_motion_r2_o5_original_4_A0_A1_A2_A3',['A0','A1','A2','A3'])]


def load(ledger):
    audit=json.loads(Path(ledger['baseline_audit']).read_text())
    manifest=json.loads(Path(ledger['motion_manifest']).read_text())
    source=Path(ledger['motion_csv'])
    assert manifest['status']=='COMPLETE' and manifest['target']=='state_delta_h5' and manifest['horizon']==5
    assert manifest['cross_arm_identical_episodes_masks_targets_and_splits']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['csv_sha256']
    for name,sha in audit['historical_csv_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    df=pd.read_csv(source)
    expected={(a,m,d) for a,_ in NEW_METHODS for m in ['ridge','mlp'] for d,_ in original.GROUPS}
    assert len(df)==64 and set(zip(df.method,df.probe,df.action_dim))==expected
    assert set(df.target)=={'state_delta_h5'} and np.isfinite(df.r2).all()
    assert set(df.checkpoint_step)=={30000}
    for arm,_ in NEW_METHODS:
        row=ledger['arms'][arm];entry=manifest['arms'][arm]
        label=json.loads((Path(row['labeled_root'])/'label_manifest.json').read_text())
        assert Path(label['source_dataset_root']).resolve()==Path(audit['source_dataset']).resolve()
        assert row['seed']==1003 and row['dataset']=='original4'
        assert int(entry['teacher_job'])==row['teacher_job']
        assert entry['checkpoint']==row['checkpoint'] and entry['labeled_root']==row['labeled_root']
        for key in ['n_train','n_test','row_keys_sha256','valid_mask_sha256','target_sha256','train_keys_sha256','test_keys_sha256']:
            assert entry[key]==audit['original4_o5'][key],(arm,key)
        for probe in ['mlp','ridge']:
            selected=df[(df.method==arm)&(df.probe==probe)]
            assert (selected.teacher_job==row['teacher_job']).all()
            assert (selected.source_labeled_dataset==row['labeled_root']).all()
            mean=selected.loc[selected.action_dim>=0,'r2'].mean()
            assert np.isclose(mean,selected.loc[selected.action_dim==-1,'r2'].iloc[0],rtol=0,atol=1e-12)
            assert np.isclose(mean,entry[probe]['mean_r2'],rtol=0,atol=1e-12)
            assert np.allclose(selected.set_index('action_dim').loc[list(range(7)),'r2'],entry[probe]['r2'],rtol=0,atol=1e-12)
            assert (selected.n_train==entry['n_train']).all() and (selected.n_test==entry['n_test']).all()
    new=df[df.probe=='mlp'].set_index(['method','action_dim']).r2
    return pd.concat([original.load(),new]),audit


def make_comparison(scores,arms):
    methods=list(original.METHODS)+[x for x in NEW_METHODS if x[0] in arms]
    assert len(methods)==4+len(arms) and len(methods) in [6,8]
    # Patch only runtime parameters of the original renderer; defaults and source
    # of the historical four-method figure remain unchanged.
    with patch.multiple(original,METHODS=methods,COLORS=COLORS,BAR_W=.75/len(methods)):
        with patch.object(original.style,'COL_WIDTH',original.style.TEXT_WIDTH):
            fig=original.make_figure(scores)
    ax=fig.axes[0];height=fig.get_size_inches()[1]
    # Two legend rows above; retain the original renderer's bottom margin.
    fig.subplots_adjust(top=1-.62/height,bottom=.29/height)
    values=[scores[(method,d)] for method,_ in methods for d,_ in original.GROUPS]
    minimum=min(values)
    lower=min(0,np.floor((minimum-.025)*10)/10) if minimum<0 else 0
    ax.set_ylim(lower,max(1,float(max(values))+.025))
    ticks=[0,.5,1]
    if lower<0:
        ticks=[lower]+ticks
        ax.axhline(0,color=original.style.INK_MUTED,linewidth=.5)
    ax.set_yticks(ticks)
    assert not fig.texts, 'Keep methodological notes in expe_docu.md, outside the figure'
    assert len(ax.patches)==len(methods)*len(original.GROUPS)
    assert np.allclose([bar.get_height() for bar in ax.patches],values,rtol=0,atol=1e-14)
    assert ax.get_ylim()[0]<=min(values)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    artists=[ax.get_legend(),ax.xaxis.label,ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()+fig.texts
    for artist in artists:
        box=artist.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,'Clipped figure text'
    assert not ax.get_legend().get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)),'Legend overlaps bars'
    return fig


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--ledger',type=Path,required=True)
    ap.add_argument('--overwrite',action='store_true',help='Regenerate only the two named PDF/PNG pairs')
    args=ap.parse_args()
    ledger=json.loads(args.ledger.read_text());scores,_=load(ledger)
    if not args.overwrite:
        for stem,_ in SPECS:
            for ext in ['pdf','png']:assert not (original.style.FIG_DIR/(stem+'.'+ext)).exists(), 'Use --overwrite to regenerate figures'
    original.style.apply_style()
    print(f'Science style: {original.style.__file__}; stack={original.style.STYLE_STACK}',flush=True)
    rendered=[(stem,arms,make_comparison(scores,arms)) for stem,arms in SPECS]
    for stem,arms,fig in rendered:
        print(stem,'historical +',arms,flush=True)
        print(scores.loc[[m for m,_ in original.METHODS]+arms].to_string(),flush=True)
        original.style.save(fig,stem)
        from matplotlib import image as mpimg
        png=original.style.FIG_DIR/(stem+'.png');pdf=original.style.FIG_DIR/(stem+'.pdf')
        pixels=mpimg.imread(png)
        assert pixels.shape[1]>1500 and pixels.std()>.01
        assert pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size>1000
    print('Renderer checks passed for both PDF/PNG pairs; human visual review remains pending.',flush=True)


if __name__=='__main__':main()
