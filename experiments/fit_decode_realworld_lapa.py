# LAPA old4 real-world probe.
#
# Adapted from experiments/fit_decode_realworld.py in this repository.
# Probe definition/training recipe is intentionally unchanged:
#
#   abs      = stored DK1 action
#   delta_h5 = observation.state[t+5] - observation.state[t]
#
# Ridge(alpha=1) and MLP [512,256], episode-level 20% holdout,
# seed 42, latent StandardScaler fitted on training rows.
#
# Difference from the original script:
#   - reads two explicitly supplied labeled exports:
#       LAPA-init and LAPA-from-scratch
#   - does not write decoded trajectory parquet files, because we only need
#     the per-dimension probing figure.
#
# Output schema matches the original probe_realworld_local.csv.

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


H = 5
DIMS = ["x", "y", "z", "wx", "wy", "wz", "grip"]

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


class TorchMLP:
    """
    Exact MLP recipe from experiments/fit_decode_realworld.py:
    [512,256] ReLU, Adam 1e-3, batch 2048, <=80 epochs,
    10% internal validation, patience 8.
    """

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        n = len(X)
        perm = rng.permutation(n)
        nv = n // 10

        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)

        self.ym = Yt[perm[nv:]].mean(0)
        self.ys = Yt[perm[nv:]].std(0) + 1e-6
        Yn = (Yt - self.ym) / self.ys

        tr = perm[nv:]
        va = perm[:nv]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, Y.shape[1]),
        )

        opt = torch.optim.Adam(self.net.parameters(), 1e-3)

        best = 1e9
        bad = 0
        state = None

        for ep in range(80):
            self.net.train()
            p = torch.tensor(rng.permutation(tr))

            for i in range(0, len(p), 2048):
                b = p[i : i + 2048]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    self.net(Xt[b]), Yn[b]
                )
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(
                    self.net(Xt[va]), Yn[va]
                ).item()

            if vl < best - 1e-5:
                best = vl
                bad = 0
                state = {
                    k: v.clone()
                    for k, v in self.net.state_dict().items()
                }
            else:
                bad += 1
                if bad >= 8:
                    break

        self.net.load_state_dict(state)
        self.epochs = ep + 1
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return (
                self.net(torch.tensor(X, dtype=torch.float32))
                * self.ys
                + self.ym
            ).numpy()


MK = {
    "ridge": lambda: Ridge(alpha=1.0),
    "mlp": lambda: TorchMLP(0),
}

PROBES = os.environ.get("PROBES", "ridge,mlp").split(",")


def run_export(method, root):
    print()
    print("=" * 80)
    print(method)
    print("root:", root)
    print("=" * 80, flush=True)

    # Each export already carries the exact source episode metadata used
    # when its latent labels were generated.
    src = pd.read_parquet(
        root / "meta" / "source_episodes.parquet"
    )[["episode_index", "task", "source_kind"]]

    table = (
        ds.dataset(root / "data", format="parquet")
        .to_table(
            columns=[
                "episode_index",
                "frame_index",
                "action",
                "observation.state",
                "latent_labels.continuous_vector_latents",
                "latent_labels.valid",
            ]
        )
        .to_pandas()
        .merge(src, on="episode_index")
        .sort_values(["episode_index", "frame_index"])
        .reset_index(drop=True)
    )

    Z = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(x, dtype=np.float32).ravel()
                    for x in a
                ]
            )
            for a in table[
                "latent_labels.continuous_vector_latents"
            ].values
        ]
    )

    A = np.stack(table.action.values)
    S = np.stack(table["observation.state"].values)

    valid = table["latent_labels.valid"].values.astype(bool)
    eps = table.episode_index.values

    # Paper real-world motion target:
    # actual EE displacement over the LAM horizon H=5.
    same = np.roll(eps, -H) == eps
    D = np.roll(S, -H, axis=0) - S

    targets = {
        "abs": (A, valid),
        "delta_h5": (D, valid & same),
    }

    # Same episode split as the official script.
    rng = np.random.RandomState(42)

    robot_valid = (
        (table.source_kind.values == "robot_3cam")
        & valid
    )

    robot_eps = np.unique(eps[robot_valid])

    test_eps = rng.choice(
        robot_eps,
        int(0.2 * len(robot_eps)),
        replace=False,
    )

    print("robot episodes:", len(robot_eps), flush=True)

    rows = []

    for target_name, (Y, ok) in targets.items():
        robot = (
            (table.source_kind.values == "robot_3cam")
            & ok
        )

        test = robot & np.isin(eps, test_eps)
        train = robot & ~np.isin(eps, test_eps)

        # Same as official fit_decode_realworld.py:
        # standardize latent using training rows only.
        scaler = StandardScaler().fit(Z[train])

        print(
            f"\ntarget={target_name}: "
            f"n_train={int(train.sum())}, "
            f"n_test={int(test.sum())}",
            flush=True,
        )

        for probe_name in PROBES:
            t0 = time.time()

            model = MK[probe_name]().fit(
                scaler.transform(Z[train]),
                Y[train],
            )

            pred = model.predict(
                scaler.transform(Z[test])
            )

            r2 = [
                r2_score(
                    Y[test][:, d],
                    pred[:, d],
                )
                for d in range(7)
            ]

            print(
                f"{method:14s} "
                f"{target_name:9s} "
                f"{probe_name:5s} "
                f"{np.round(r2, 3)} "
                f"mean {np.mean(r2):.3f} "
                f"xyz {np.mean(r2[:3]):.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

            for d, score in enumerate(r2):
                rows.append(
                    dict(
                        method=method,
                        target=target_name,
                        probe=probe_name,
                        action_dim=d,
                        r2=score,
                        n_train=int(train.sum()),
                        n_test=int(test.sum()),
                    )
                )

            rows.append(
                dict(
                    method=method,
                    target=target_name,
                    probe=probe_name,
                    action_dim=-1,
                    r2=float(np.mean(r2)),
                    n_train=int(train.sum()),
                    n_test=int(test.sum()),
                )
            )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scratch-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    from pathlib import Path

    init_root = Path(args.init_root).resolve()
    scratch_root = Path(args.scratch_root).resolve()
    output = Path(args.output).resolve()

    for root in (init_root, scratch_root):
        assert (root / "meta" / "info.json").is_file()
        assert (root / "meta" / "source_episodes.parquet").is_file()
        assert (root / "data").is_dir()

    rows = []

    rows.extend(
        run_export(
            "lapa_init",
            init_root,
        )
    )

    rows.extend(
        run_export(
            "lapa_scratch",
            scratch_root,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print()
    print("saved:", output)

    print()
    print("=== MLP / delta_h5 ===")

    x = df[
        (df.probe == "mlp")
        & (df.target == "delta_h5")
    ].copy()

    names = {
        -1: "Mean",
        0: "dx",
        1: "dy",
        2: "dz",
        3: "dtheta_x",
        4: "dtheta_y",
        5: "dtheta_z",
        6: "grip",
    }

    x["dimension"] = x.action_dim.map(names)

    print(
        x[
            ["method", "dimension", "r2"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
