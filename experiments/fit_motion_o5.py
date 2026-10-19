"""Isolated newtasks A0-A3 motion probe; preserves the original DK1 recipe.

Uses the unchanged TorchMLP from fit_decode_realworld_lapa.py (the parameterized
copy of fit_decode_realworld.py). Only state_delta_h5 is scored; no decode refits.
Fails before fitting if arms disagree on episodes, valid frames, states or split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from fit_decode_realworld_lapa import TorchMLP

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK/'repo/code/multiview_spatial_o5'))
from posttrain import audit, audit_policy

NAMES = ['ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz', 'ee.gripper_pos']
DIMS = ['dx', 'dy', 'dz', 'dtheta_x', 'dtheta_y', 'dtheta_z', 'gripper']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(row):
    root = Path(row['labeled_root'])
    checked = audit(root)
    info = json.loads((root/'meta/info.json').read_text())
    assert info['features']['observation.state']['names'] == NAMES
    src = pd.read_parquet(root/'meta/source_episodes.parquet')
    assert not src.episode_index.duplicated().any()
    robot_eps_all = np.sort(src.loc[src.source_kind == 'robot_3cam', 'episode_index'].to_numpy())
    table = ds.dataset(root/'data', format='parquet').to_table(columns=[
        'episode_index', 'frame_index', 'observation.state',
        'latent_labels.continuous_vector_latents', 'latent_labels.valid'],
        filter=ds.field('episode_index').isin(robot_eps_all.tolist())).to_pandas()
    table = table.sort_values(['episode_index', 'frame_index']).reset_index(drop=True)
    keys = table[['episode_index', 'frame_index']].to_numpy(dtype=np.int64)
    assert not table.duplicated(['episode_index', 'frame_index']).any()
    eps, frames = keys.T
    assert np.array_equal(np.unique(eps), robot_eps_all)
    for ep in robot_eps_all:
        f = frames[eps == ep]
        assert np.array_equal(f, np.arange(len(f))), 'Non-contiguous original frame ordering'
    state = np.stack(table['observation.state'].to_numpy())
    assert state.shape == (len(table), 7) and np.isfinite(state).all()
    # Exact nested-latent decoder from the original DK1 probe.
    z = np.stack([np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in a])
                  for a in table['latent_labels.continuous_vector_latents'].values])
    assert z.shape == (len(table), 8) and z.dtype == np.float32
    valid = table['latent_labels.valid'].values.astype(bool)
    same = np.roll(eps, -5) == eps
    # Exactly the original subtraction; no axis, angular or unit transformation.
    target = np.roll(state, -5, axis=0) - state
    ok = valid & same
    robot_eps = np.unique(eps[valid])
    test_eps = np.random.RandomState(42).choice(robot_eps, int(0.2*len(robot_eps)), replace=False)
    test = ok & np.isin(eps, test_eps)
    train = ok & ~np.isin(eps, test_eps)
    assert np.isfinite(z[ok]).all() and train.any() and test.any()
    details = dict(export_audit=checked, robot_episodes=robot_eps_all.tolist(),
                   train_episodes=np.unique(eps[train]).tolist(), test_episodes=np.unique(eps[test]).tolist(),
                   n_train=int(train.sum()), n_test=int(test.sum()),
                   valid_mask_sha256=digest(valid), row_keys_sha256=digest(keys), target_sha256=digest(target[ok]),
                   train_keys_sha256=digest(keys[train]), test_keys_sha256=digest(keys[test]),
                   std_per_dim=z[ok].std(0).tolist(), units='Original DK1 state units, unchanged; R2 dimensionless')
    return dict(z=z, y=target, valid=valid, keys=keys, state=state, train=train, test=test, details=details)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--arm', choices=['A0', 'A1', 'A2', 'A3'], help='Evaluate one ledger arm only')
    args = ap.parse_args()
    ledger = json.loads(args.ledger.read_text())
    if args.arm:
        ledger['arms'] = {args.arm: ledger['arms'][args.arm]}
    out = Path(ledger['motion_csv'])
    manifest_path = Path(ledger['motion_manifest'])
    assert not out.exists() and not manifest_path.exists(), 'Never overwrite an experiment'
    manifest = dict(status='VALIDATING', target='state_delta_h5', horizon=5,
                    split_seed=42, split='RandomState.choice 20% of unique valid robot episodes',
                    mlp_seed=0, mlp_reference='experiments/fit_decode_realworld_lapa.py:TorchMLP',
                    latent_scaling='StandardScaler fit on training rows only', ridge_alpha=1,
                    state_dimensions=NAMES, baseline_comparison=ledger['baseline_comparison'], arms={})
    arrays = {}
    try:
        for arm, row in ledger['arms'].items():
            item = load(row)
            if arrays:
                ref = arrays['A0']
                for key in ['keys', 'state', 'valid', 'train', 'test']:
                    assert np.array_equal(item[key], ref[key]), f'{arm}: {key} mismatch; no intersection or resplit allowed'
            arrays[arm] = item
            manifest['arms'][arm] = dict(item['details'], teacher_job=row['teacher_job'], checkpoint=row['checkpoint'], labeled_root=row['labeled_root'])
        # The job waits afterany on gates, so failed gates do not block motion results.
        for arm, row in ledger['arms'].items():
            if (Path(row['policy_root'])/'summary.json').exists():
                manifest['arms'][arm]['policy_validation'] = audit_policy(row, write=False)
        verified = [a['policy_validation'] for a in manifest['arms'].values() if 'policy_validation' in a]
        if verified:
            for value in verified[1:]:
                for key in ['episode_sets', 'row_keys_sha256', 'target_sha256', 'n_valid']:
                    assert value[key] == verified[0][key], 'Policy arms differ: '+key
        manifest['status'] = 'FITTING'
        manifest['cross_arm_identical_episodes_masks_targets_and_splits'] = True
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        rows = []
        for arm, item in arrays.items():
            z, y, train, test = (item[k] for k in ['z', 'y', 'train', 'test'])
            scaler = StandardScaler().fit(z[train])
            source = ledger['arms'][arm]
            for probe, model in [('ridge', Ridge(alpha=1.0)), ('mlp', TorchMLP(0))]:
                start = time.time()
                model.fit(scaler.transform(z[train]), y[train])
                pred = model.predict(scaler.transform(z[test]))
                scores = np.array([r2_score(y[test, d], pred[:, d]) for d in range(7)])
                assert np.isfinite(scores).all()
                base = dict(method=arm, teacher_job=source['teacher_job'], target='state_delta_h5', probe=probe,
                            mean_r2=float(scores.mean()), n_train=int(train.sum()), n_test=int(test.sum()),
                            train_episode_count=len(item['details']['train_episodes']), test_episode_count=len(item['details']['test_episodes']),
                            source_labeled_dataset=source['labeled_root'], checkpoint_step=30000)
                for d in [-1]+list(range(7)):
                    rows.append(dict(base, action_dim=d, dim_name='mean' if d == -1 else DIMS[d], r2=float(scores.mean() if d == -1 else scores[d])))
                print(arm, probe, scores, 'mean', scores.mean(), 'seconds', time.time()-start, flush=True)
                manifest['arms'][arm][probe] = dict(epochs=getattr(model, 'epochs', None), mean_r2=float(scores.mean()), r2=scores.tolist())
                manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        frame = pd.DataFrame(rows)
        assert len(frame) == len(ledger['arms'])*2*8
        # Exclusive creation and readback: source CSV is immutable once complete.
        with out.open('x') as handle:
            frame.to_csv(handle, index=False)
        reread = pd.read_csv(out)
        assert len(reread) == len(frame) and np.allclose(reread.r2, frame.r2, rtol=0, atol=1e-14)
        manifest['status'] = 'COMPLETE'
        manifest['csv_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    except Exception as exc:
        manifest.update(status='ERROR', error=repr(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
