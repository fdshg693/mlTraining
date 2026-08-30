# test_sizeとrandom_stateを振って、同じモデルでも分割の仕方次第で評価指標が
# どれだけブレるかを観察する実験
#
# 【実行前に予想】lifesatは27件しかない小データ。test_size（テストの割合）を
# 小さくして、テスト件数を減らすと、乱数シードによるR^2のブレは大きくなる？
# それとも小さくなる？

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_produce.util.csv_io import read_csv
from data_produce.util.logging_config import setup_logger

logger = setup_logger(Path(__file__).stem)

outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

lifesat = read_csv("lifesat.csv")
X = lifesat[[gdppc_col]].values
y = lifesat[lifesat_col].values

test_sizes = (0.1, 0.2, 0.3, 0.4)
seeds = range(20)

all_scores = []
tick_labels = []

for test_size in test_sizes:
    scores = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        model = LinearRegression().fit(X_train, y_train)
        scores.append(r2_score(y_test, model.predict(X_test)))

    scores = np.array(scores)
    n_test = len(X_test)
    all_scores.append(scores)
    tick_labels.append(f"{test_size}\n(n_test={n_test})")

    logger.info(
        f"test_size={test_size:.1f} (test={n_test}件): "
        f"R^2 平均={scores.mean():.3f} 標準偏差={scores.std():.3f} "
        f"min={scores.min():.3f} max={scores.max():.3f}"
    )

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(all_scores, tick_labels=tick_labels)
ax.set_xlabel("test_size")
ax.set_ylabel("R^2 (test)")
ax.set_title(f"R^2 variance by test_size (random_state=0-{len(seeds) - 1}, n={len(seeds)} runs)")
ax.grid(True)
fig.savefig(outputs_dir / "split_seed_sweep.png", dpi=150, bbox_inches="tight")
logger.info(f"saved -> {outputs_dir / 'split_seed_sweep.png'}")

# 【観察メモ】test_sizeが小さい（テスト件数が少ない）ほど、1〜2件の
# 外れ値がテストに入るかどうかでR^2が大きく揺れ、標準偏差・箱ひげの幅も
# 大きくなる。27件しかないlifesatのような小データでは、1回のホールドアウト
# 検証だけでモデルの優劣を判断するのは危険 -> cross_validation.pyへ続く。
