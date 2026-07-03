import argparse
import os
import numpy as np
from model import *
from utils import *
from torch import optim
from accelerate import get_mat1, get_mat2
from sklearn.cluster import SpectralClustering


def autoencoder_block_b(adata, gene_list, n_sample, epoch, seed):
    """Original scMUG block B: ZINB autoencoder per GFM."""
    set_seed(seed)
    batchSize = int(max(min(2 ** (n_sample**0.5 // 8 + 1), 64), 4))
    mask = adata.var_names.isin(gene_list)
    input_data, target_data = adata.X[:, mask], adata.X[:, mask]
    input_dim = input_data.shape[1]
    data_loader = get_data_loader(input_data, target_data, batch_size=batchSize)
    model = Autoencoder([input_dim, 512, 128, 32]).to(device)
    criterion = ZINBLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, data_loader, epoch, criterion, optimizer)
    return get_encoded_output(model, data_loader)


def dmkcn_adapter_block_b(
    adata,
    gene_list,
    cluster_number,
    seed,
    zinb_on_counts=True,
    full_training=True,
):
    """DMKCN replacement for scMUG block B. Imported lazily to keep AE path usable."""
    from dmkcn.integration import dmkcn_block_b

    return dmkcn_block_b(
        adata,
        gene_list,
        n_clusters=cluster_number,
        d=32,
        seed=seed,
        full_training=full_training,
        zinb_on_counts=zinb_on_counts,
    )


def run():
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", default="muraro", type=str)
    parser.add_argument("--cluster_number", default=None, type=int)
    parser.add_argument(
        "--seeds",
        default="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000",
        type=str,
    )
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--n_gfm", default=5, type=int)
    parser.add_argument("--cutoffs", default="0.14,0.14,0.15,0.14,0.14", type=str)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--n_neighbour", default=3, type=int)
    parser.add_argument("--kmeans_times", default=20, type=int)
    parser.add_argument("--red_global", type=str)
    parser.add_argument("--red_local", type=str)
    parser.add_argument(
        "--block-b",
        default="dmkcn",
        choices=["autoencoder", "dmkcn"],
        help="Block B representation generator. 'autoencoder' is original scMUG; 'dmkcn' is the adapter branch.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        type=str,
        help="Suffix used for output files. Defaults to the selected --block-b.",
    )
    parser.add_argument(
        "--dmkcn-zinb-on-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only for --block-b dmkcn. True uses raw counts as ZINB target; false mirrors scMUG scaled-target behavior.",
    )
    parser.add_argument(
        "--dmkcn-full-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only for --block-b dmkcn. Disable to run DMKCN pretraining only.",
    )

    args = parser.parse_args()

    dbname = args.dataset
    seeds = [int(_) for _ in args.seeds.split(",")]
    repeat = args.repeat
    n_gfm = args.n_gfm
    cluster_number = args.cluster_number
    cutoffs = [float(_) for _ in args.cutoffs.split(",")]
    epoch = args.epoch
    n_neighbour = args.n_neighbour
    kmeans_times = args.kmeans_times
    red_global = args.red_global
    red_local = args.red_local
    block_b = args.block_b
    output_tag = args.output_tag or block_b

    expr_df, cell_type = load_data(dbname)
    print(f"\nDatabase: {dbname}\tCells: {expr_df.shape[0]}\tGenes: {expr_df.shape[1]}")
    print(f"Block B: {block_b}\tOutput tag: {output_tag}")
    expr_df = expr_df.astype(float)
    adata = preprocess(expr_df=expr_df, cell_type=cell_type, highly_genes=8000)
    y = lab2fac(adata.obs["cell_type"].to_numpy())
    n_sample = adata.X.shape[0]
    if cluster_number is None:
        cluster_number = len(set(y))

    predictions = []
    latents = []
    os.makedirs("./outputs", exist_ok=True)
    f = open(f"./outputs/{dbname}s_{output_tag}.txt", "w", encoding="utf-8")

    for seed in seeds:
        print(f"\nSeed: {seed}\n")
        latent_val = None
        for i, t in enumerate(cutoffs):
            print(f"GFM: {i + 1}")
            with open(f"./GFMs/{dbname}/{i + 1}.txt", "r") as fg:
                gfm = set(
                    [
                        s.replace("\n", "").replace("\t", "")
                        for s in fg.readlines()
                        if s[0] != "#"
                    ]
                )
            gfm = list(gfm & set(adata.var.index))
            gene_list = extend_gfm(adata, gfm, t, d=3)

            if block_b == "autoencoder":
                latent = autoencoder_block_b(
                    adata=adata,
                    gene_list=gene_list,
                    n_sample=n_sample,
                    epoch=epoch,
                    seed=seed,
                )
            elif block_b == "dmkcn":
                set_seed(seed)
                latent = dmkcn_adapter_block_b(
                    adata=adata,
                    gene_list=gene_list,
                    cluster_number=cluster_number,
                    seed=seed,
                    zinb_on_counts=args.dmkcn_zinb_on_counts,
                    full_training=args.dmkcn_full_training,
                )
            else:
                raise ValueError(f"Unknown --block-b: {block_b}")

            latent = latent.reshape((latent.shape[0], 1, -1))
            if i == 0:
                latent_val = latent
            else:
                latent_val = np.concatenate((latent_val, latent), axis=1)

        latents.append(latent_val)

        # local feature
        dist = np.zeros(shape=(n_gfm, n_sample, n_sample))
        for c in range(n_gfm):
            X = latent_val[:, c, :]

            if not np.isfinite(X).all():
                print(f"[WARNING] NaN/Inf detected before local reducer, GFM={c + 1}")
                print("NaN count:", np.isnan(X).sum())
                print("Inf count:", np.isinf(X).sum())
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            latent_val_c = reducer(red_local)(X)

            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    dist[c, i, j] = dist[c, j, i] = np.linalg.norm(
                        latent_val_c[i] - latent_val_c[j]
                    )

        neighbourDist = np.array(
            [
                np.array(
                    [
                        np.sum(
                            dis[
                                i,
                                np.argpartition(dis[i], n_neighbour + 1)[
                                    1 : n_neighbour + 1
                                ],
                            ]
                        )
                        / n_neighbour
                        for i in range(n_sample)
                    ]
                )
                for dis in dist
            ]
        )

        neighbourDistScore = (
            np.array(
                [
                    np.array(
                        [
                            1
                            / np.log(
                                np.var(
                                    dis[
                                        i,
                                        np.argpartition(dis[i], n_neighbour + 1)[
                                            1 : n_neighbour + 1
                                        ],
                                    ]
                                )
                                + np.exp(1)
                            )
                            for i in range(n_sample)
                        ]
                    )
                    for dis in dist
                ]
            )
            ** 0.5
        )

        mat2 = get_mat2(n_sample, neighbourDist, dist, neighbourDistScore)

        for r in range(repeat):
            print(f"\nRound {r}")
            # global feature
            pred = np.zeros(shape=(kmeans_times, n_gfm, latent_val.shape[0])).astype(
                int
            )
            score = np.zeros(shape=(kmeans_times, n_gfm, latent_val.shape[0]))
            for c in range(n_gfm):
                z = latent_val[:, c, :]
                z = z.reshape(z.shape[0], -1)
                if not np.isfinite(z).all():
                    print(
                        f"[WARNING] NaN/Inf detected before global reducer, GFM={c + 1}"
                    )
                    print("NaN count:", np.isnan(z).sum())
                    print("Inf count:", np.isinf(z).sum())
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                z = reducer(red_global)(z)

                for t in range(kmeans_times):
                    pred_z, score_z = c_kmeans(
                        z, cluster_number, n_init=10, random_state=None
                    )
                    pred[t, c, :] = pred_z
                    score_z = np.sort(score_z, axis=1)
                    denom = score_z[:, 1] + score_z[:, 0]
                    denom = np.where(denom == 0, 1e-12, denom)
                    score[t, c, :] = (
                        ((score_z[:, 1] - score_z[:, 0]) / denom) ** 0.5
                        / kmeans_times
                        / n_gfm
                    )

            mat1 = get_mat1(
                pred, n_sample, kmeans_times, n_gfm, cluster_number, score, 8
            )
            mat1 = mat1 / np.mean(mat1)

            for alpha, beta in [
                (0, 1),
                (0.001, 1),
                (0.01, 1),
                (0.1, 1),
                (1, 1),
                (1, 0.1),
                (1, 0.01),
                (1, 0.001),
                (1, 0),
            ]:
                mat = mat1 * alpha + mat2 * beta
                Spec = SpectralClustering(
                    n_clusters=cluster_number, random_state=None, affinity="precomputed"
                )
                labels = Spec.fit_predict(mat)
                print(
                    f"dbname:{dbname}\tround:{seed}\talpha:{round(alpha, 3)}\tbeta:{round(beta, 3)}\t",
                    end="",
                )
                _ = benchmark(y, labels)
                p12 = labels
                predictions.append(p12)
                f.write(
                    f"dbname:{dbname}\tround:{seed}\talpha:{alpha}\tbeta:{beta}\t{benchmark(y, p12, False)}\n"
                )

    f.close()
    joblib.dump(
        predictions, f"./outputs/{dbname}_pred_label_{output_tag}.joblib"
    )  # predicted labels
    joblib.dump(
        latents, f"./outputs/{dbname}_latents_{output_tag}.joblib"
    )  # latent features for different GFMs


if __name__ == "__main__":
    run()
