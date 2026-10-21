"""Two new ALL-O5 figures; reuse the existing six/eight-method Science renderer.

All methods: original4, O5, teacher seed1003. Refuses missing/unmatched data.
Historical O159 CSV loaders are never called. LaTeX: figure*, width=\\textwidth.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plot_motion_original4 as renderer

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'repo/code/multiview_spatial_o5'))
from all_o5 import read, sha, MATCH_KEYS
from all_o5_verify import verify_motion, verify_policy


def load(ledger):
    new=ledger['groups']['historical_original4_seed1003']
    existing=read(WORK/'repo/code/multiview_spatial_o5/original4_jobs.json')
    reference=read(existing['motion_manifest'])
    ref=reference['arms']['A0']
    frames=[]
    sources={}
    assert new['dataset']=='original4' and new['seed']==1003
    assert set(new['arms'])=={'ours_single','ours_multi','lapa_scratch','lapa_init'}
    groups=[new,dict(existing,dataset='original4',seed=1003)]
    for group in groups:
        assert set(group['arms'])==(set(new['arms']) if group is new else {'A0','A1','A2','A3'})
        for arm,row in group['arms'].items():
            assert row['dataset']=='original4' and row['seed']==1003
            if group is new:
                assert row['provenance_status']=='VERIFIED'
            _,proof=verify_motion(row,group,arm,ref)
            verify_policy(row,group,ref)
            sources.update(proof['sources_sha256'])
        frame=pd.read_csv(group['motion_csv'])
        frames.append(frame.loc[frame.probe=='mlp',['method','action_dim','r2']])
    result=pd.concat(frames,ignore_index=True)
    assert len(result)==64 and not result.duplicated(['method','action_dim']).any()
    return result.set_index(['method','action_dim']).r2,sources


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--ledger',required=True,type=Path)
    args=ap.parse_args()
    ledger=read(args.ledger)
    specs=[(stem+'_ALLo5',arms) for stem,arms in renderer.SPECS]
    manifest=Path(__file__).resolve().parent/'results/all_o5_original4_plot_manifest.json'
    assert not manifest.exists()
    for stem,_ in specs:
        for ext in ['pdf','png']:
            assert not (renderer.original.style.FIG_DIR/(stem+'.'+ext)).exists()
    scores,sources=load(ledger)
    renderer.original.style.apply_style()
    outputs={}
    for stem,arms in specs:
        fig=renderer.make_comparison(scores,arms)
        # make_comparison checks method count, exact measurements, negative values,
        # all text bounds and legend/bar overlap using the original layout.
        assert len(fig.axes[0].patches)==(4+len(arms))*8
        renderer.original.style.save(fig,stem)
        from matplotlib import image as mpimg
        png=renderer.original.style.FIG_DIR/(stem+'.png')
        pdf=renderer.original.style.FIG_DIR/(stem+'.pdf')
        pixels=mpimg.imread(png)
        assert pixels.shape[1]>1500 and pixels.std()>.01
        assert pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size>1000
        outputs.update({str(p):sha(p) for p in [png,pdf]})
        print(stem,'\n',scores.to_string(),flush=True)
    with manifest.open('x') as out:
        json.dump(dict(status='COMPLETE',dataset='original4',teacher_seed=1003,offsets=[5],
                       sources_sha256=sources,outputs_sha256=outputs,
                       renderer='plot_motion_original4.make_comparison; plot_realworld_probe_all4.make_figure',
                       checks=['48/64 bars','original ordering and colors','saved measurements',
                               'negative values visible','text bounds','legend overlap','PDF/PNG readback'],
                       human_visual_review='PENDING'),out,indent=2)
        out.write('\n')


if __name__=='__main__':
    main()
