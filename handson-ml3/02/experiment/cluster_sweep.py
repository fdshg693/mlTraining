"""ClusterSimilarityのn_clusters・gammaを変え、緯度経度の類似度特徴量が
空間的にどう変わるかを観察する実験。

09_tune_model.pyではn_clustersをCV RMSEだけで最適化した（結果は数値でしか
見えない）。ここでは同じパラメータが地図上の類似度分布としてどう見えるかを
可視化し、数値の裏側にある形を確認する。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "learn"))
from common import ClusterSimilarity, load_features_and_labels  # noqa: E402
from data_produce.util.logging_config import setup_logger  # noqa: E402

RANDOM_STATE = 42
N_CLUSTERS_VALUES = (3, 10, 30, 50)
GAMMA_VALUES = (1.0, 0.1, 0.05, 0.01)
FIXED_GAMMA = 1.0  # n_clusters sweepで固定するgamma（build_full_preprocessing()と同じ既定値）
GRID_RESOLUTION = 150


def build_coord_grid(
    coords: np.ndarray, resolution: int = GRID_RESOLUTION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """緯度・経度の範囲を覆うメッシュグリッドを作る（等高線ヒートマップ描画用）。"""

    lat_lin = np.linspace(coords[:, 0].min(), coords[:, 0].max(), resolution)
    lon_lin = np.linspace(coords[:, 1].min(), coords[:, 1].max(), resolution)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return lon_grid, lat_grid, grid_points


def compute_n_clusters_results(coords: np.ndarray) -> dict[int, dict]:
    """gammaを固定し、n_clustersごとにClusterSimilarityを学習・変換する。"""

    results = {}
    for n_clusters in N_CLUSTERS_VALUES:
        cluster_simil = ClusterSimilarity(
            n_clusters=n_clusters, gamma=FIXED_GAMMA, random_state=RANDOM_STATE
        )
        similarities = cluster_simil.fit_transform(coords)
        results[n_clusters] = {
            "cluster_simil": cluster_simil,
            "max_similarity": similarities.max(axis=1),
        }
    return results


def compute_gamma_results(coords: np.ndarray, grid_points: np.ndarray) -> dict[float, dict]:
    """n_clusters=1に固定し、gammaごとにClusterSimilarityを学習・変換する。

    1クラスタに固定することで、他クラスタとの競合なしにgammaだけの効果
    （類似度が届く距離）を観察できる。
    """

    results = {}
    for gamma in GAMMA_VALUES:
        cluster_simil = ClusterSimilarity(n_clusters=1, gamma=gamma, random_state=RANDOM_STATE)
        cluster_simil.fit(coords)
        similarity_at_data = cluster_simil.transform(coords).ravel()
        similarity_grid = cluster_simil.transform(grid_points).reshape(-1)
        results[gamma] = {
            "cluster_simil": cluster_simil,
            "similarity_at_data": similarity_at_data,
            "similarity_grid": similarity_grid,
        }
    return results


def plot_n_clusters_sweep(
    coords: np.ndarray, population: np.ndarray, results: dict[int, dict], outputs_dir: Path
) -> Path:
    """gamma固定・n_clusters違いの「地区ごとの最大クラスタ類似度」を並べる。"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, n_clusters in zip(axes.ravel(), N_CLUSTERS_VALUES):
        cluster_simil = results[n_clusters]["cluster_simil"]
        max_similarity = results[n_clusters]["max_similarity"]

        scatter = ax.scatter(
            coords[:, 1], coords[:, 0], c=max_similarity, s=population / 100,
            cmap="jet", alpha=0.6, vmin=0, vmax=1,
        )
        ax.scatter(
            cluster_simil.kmeans_.cluster_centers_[:, 1],
            cluster_simil.kmeans_.cluster_centers_[:, 0],
            marker="X", color="black", s=120, label="Cluster centers",
        )
        ax.set_title(f"n_clusters={n_clusters} (gamma={FIXED_GAMMA})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        fig.colorbar(scatter, ax=ax, label="Max cluster similarity")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()

    path = outputs_dir / "n_clusters_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gamma_sweep(
    lon_grid: np.ndarray, lat_grid: np.ndarray, results: dict[float, dict], outputs_dir: Path
) -> Path:
    """n_clusters=1固定・gamma違いの「類似度の効く範囲」を等高線で並べる。"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, gamma in zip(axes.ravel(), GAMMA_VALUES):
        cluster_simil = results[gamma]["cluster_simil"]
        similarity_grid = results[gamma]["similarity_grid"].reshape(lat_grid.shape)

        contour = ax.contourf(
            lon_grid, lat_grid, similarity_grid, levels=20, cmap="viridis", vmin=0, vmax=1
        )
        ax.contour(
            lon_grid, lat_grid, similarity_grid, levels=[0.5], colors="white", linewidths=2,
        )
        center = cluster_simil.kmeans_.cluster_centers_[0]
        ax.scatter(center[1], center[0], marker="X", color="red", s=150, label="Center")
        ax.set_title(f"gamma={gamma} (n_clusters=1)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(contour, ax=ax, label="RBF similarity")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()

    path = outputs_dir / "gamma_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup_logger(Path(__file__).stem)
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    housing, _housing_labels = load_features_and_labels()
    coords = housing[["latitude", "longitude"]].to_numpy()
    population = housing["population"].to_numpy()
    logger.info(f"訓練セット: {housing.shape[0]:,}行")

    logger.info(f"\n=== n_clusters sweep（gamma={FIXED_GAMMA}固定） ===")
    n_clusters_results = compute_n_clusters_results(coords)
    n_clusters_path = plot_n_clusters_sweep(coords, population, n_clusters_results, outputs_dir)
    logger.info(f"保存しました: {n_clusters_path}")
    for n_clusters in N_CLUSTERS_VALUES:
        max_similarity = n_clusters_results[n_clusters]["max_similarity"]
        logger.info(
            f"n_clusters={n_clusters:>2}: 最大類似度の平均={max_similarity.mean():.3f}, "
            f"標準偏差={max_similarity.std():.3f}"
        )
    logger.info(
        "n_clustersを増やすとクラスタ中心が密になり、どの地区も近くの中心を"
        "持ちやすくなるため、最大類似度の平均は上がり、地区間のばらつき"
        "（標準偏差）は縮む。つまり特徴量としては全体的に『効きやすく』なる一方、"
        "地区どうしを区別する情報量はクラスタ数を増やすほど下がっていく。"
    )

    logger.info("\n=== gamma sweep（n_clusters=1固定） ===")
    lon_grid, lat_grid, grid_points = build_coord_grid(coords)
    gamma_results = compute_gamma_results(coords, grid_points)
    gamma_path = plot_gamma_sweep(lon_grid, lat_grid, gamma_results, outputs_dir)
    logger.info(f"保存しました: {gamma_path}")
    for gamma in GAMMA_VALUES:
        similarity_at_data = gamma_results[gamma]["similarity_at_data"]
        effective_share = (similarity_at_data >= 0.5).mean()
        logger.info(
            f"gamma={gamma:<5}: 類似度>=0.5の地区割合={effective_share:.1%}"
        )
    small_gamma, large_gamma = min(GAMMA_VALUES), max(GAMMA_VALUES)
    small_gamma_share = (gamma_results[small_gamma]["similarity_at_data"] >= 0.5).mean()
    large_gamma_share = (gamma_results[large_gamma]["similarity_at_data"] >= 0.5).mean()
    logger.info(
        "gammaを大きくするとRBF類似度exp(-gamma * distance^2)は中心から"
        "離れるにつれ急速に0へ近づくため、『効く範囲』（類似度>=0.5の地区割合）は"
        f"gamma={small_gamma}の{small_gamma_share:.1%}から"
        f"gamma={large_gamma}の{large_gamma_share:.1%}まで大きく縮む。"
        "n_clustersがクラスタの『数』を、gammaが各クラスタの『届く距離』を"
        "制御しているとわかる。"
    )


if __name__ == "__main__":
    main()
