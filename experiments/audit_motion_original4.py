"""Read-only original4 baseline/sample audit; never refit or adjust historical R2."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK/'vpro_paper_plots'))
from fit_motion_o5 import NAMES, digest
import plot_realworld_probe_all4 as historical


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--ledger', type=Path, required=True)
    args = ap.parse_args()
    ledger = json.loads(args.ledger.read_text())
    output = Path(ledger['baseline_audit'])
    assert not output.exists(), output
    root = Path(ledger['source_dataset']['root'])
    info = json.loads((root/'meta/info.json').read_text())
    assert info['features']['observation.state']['names'] == NAMES
    src = pd.read_parquet(root/'meta/source_episodes.parquet')
    robot_eps = np.sort(src.loc[src.source_kind == 'robot_3cam', 'episode_index'].to_numpy())
    table = ds.dataset(root/'data', format='parquet').to_table(
        columns=['episode_index','frame_index','observation.state'],
        filter=ds.field('episode_index').isin(robot_eps.tolist())).to_pandas()
    table = table.sort_values(['episode_index','frame_index']).reset_index(drop=True)
    keys = table[['episode_index','frame_index']].to_numpy(dtype=np.int64)
    eps, frames = keys.T
    state = np.stack(table['observation.state'].values)
    target = np.roll(state,-5,axis=0)-state
    test_eps = np.random.RandomState(42).choice(robot_eps,int(.2*len(robot_eps)),replace=False)
    protocols = {}
    for offset in [5,9]:
        valid = np.zeros(len(keys),dtype=bool)
        valid[:-offset] = (eps[:-offset] == eps[offset:]) & (frames[offset:] == frames[:-offset]+offset)
        train = valid & ~np.isin(eps,test_eps)
        test = valid & np.isin(eps,test_eps)
        protocols[str(offset)] = dict(n_train=int(train.sum()),n_test=int(test.sum()),
            robot_episodes=robot_eps.tolist(),test_episodes=sorted(test_eps.tolist()),
            row_keys_sha256=digest(keys),valid_mask_sha256=digest(valid),
            target_sha256=digest(target[valid]),train_keys_sha256=digest(keys[train]),test_keys_sha256=digest(keys[test]))
    # Reuse the historical CSV selectors and seven-dimension mean checks verbatim.
    historical.load()
    labels = Path('/leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/latent_labels')
    baselines = {}
    for method, name in [
        ('lapa_init','dk1_original4_lapaft_sideonly_lapainit_59093801_ckpt030000_idx1'),
        ('lapa_scratch','dk1_original4_lapaft_sideonly_scratch_59094662_ckpt030000_idx1')]:
        path = labels/name
        manifest = json.loads((path/'label_manifest.json').read_text())
        assert Path(manifest['source_dataset_root']).resolve() == root.resolve()
        assert manifest['episodes'] is None
        assert (path/'meta/source_episodes.parquet').read_bytes() == (root/'meta/source_episodes.parquet').read_bytes()
        old = ds.dataset(path/'data',format='parquet').to_table(
            columns=['episode_index','frame_index','observation.state','latent_labels.valid'],
            filter=ds.field('episode_index').isin(robot_eps.tolist())).to_pandas()
        old = old.sort_values(['episode_index','frame_index']).reset_index(drop=True)
        assert np.array_equal(old[['episode_index','frame_index']].to_numpy(dtype=np.int64),keys)
        assert np.array_equal(np.stack(old['observation.state'].values),state)
        valid = old['latent_labels.valid'].values.astype(bool)
        assert digest(valid) == protocols['9']['valid_mask_sha256']
        baselines[method] = dict(labeled_root=str(path),raw_data_verified=True,
            source_dataset_verified=True,robot_episodes_verified=True,state_targets_verified=True,
            validity_offset=9,latent_offsets=manifest['delta_timestamps'],**protocols['9'])
    for method, job, stem in [('ours_single',44160743,'sharedlam3'),('ours_multi',43524549,'sharedlam2')]:
        candidate = labels/f'dk1_postproc19_{stem}_{job}_ckpt030000_idx1'
        assert not candidate.exists(), 'Raw historical DINO data became available: audit it before comparison'
        baselines[method] = dict(teacher_job=job,raw_data_verified=False,
            reason='Historical DINO raw export unavailable on Leonardo; CSV/source-code provenance alone cannot verify exact episode/frame keys.',
            expected_export=str(candidate),csv_counts=[137152,35881])
    sources = [historical.PAPER_CSV,historical.LAPA_CSV]
    report = dict(status='AUDITED_WITH_EXPLICIT_MISMATCH',perfectly_matched=False,
        source_dataset=str(root),source_info_sha256=hashlib.sha256((root/'meta/info.json').read_bytes()).hexdigest(),
        source_metadata_sha256=hashlib.sha256((root/'meta/source_episodes.parquet').read_bytes()).hexdigest(),
        target='state[t+5]-state[t]',state_dimensions=NAMES,split_seed=42,
        methodology='Original TorchMLP/Adam/StandardScaler/Ridge and seven-dimension unweighted mean; historical CSV scores unchanged.',
        original4_o5=protocols['5'],historical_h9=protocols['9'],baselines=baselines,
        historical_csv_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        mismatch='Historical h9 validity excludes four more tail frames per episode than new o5 h5 validity; no intersection, reweighting or rescaling applied. Historical DINO raw episode/frame keys remain unverified.')
    with output.open('x') as handle:
        handle.write(json.dumps(report,indent=2)+'\n')
    print('BASELINE_AUDIT',report['status'],report['mismatch'],flush=True)
    print('native_o5_counts',protocols['5']['n_train'],protocols['5']['n_test'],flush=True)


if __name__ == '__main__':
    main()
