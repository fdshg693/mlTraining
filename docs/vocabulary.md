# 用語集

## cross_val_score

`sklearn.model_selection.cross_val_score` は、k分割交差検証（k-fold cross validation）を実行するための関数。
データをk個に分割し、「k-1個で学習 → 残り1個で評価」をk回繰り返して、fold（分割）ごとの評価スコアを配列で返す。

1回だけの`train_test_split`によるホールドアウト検証は、分割の仕方（乱数シード）次第でスコアがブレやすいが、
`cross_val_score`はk回の評価を平均・標準偏差にまとめることで、より安定した汎化性能の見積もりが得られる。

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
# scores は fold ごとのスコアが入った長さ5の配列
```

主な引数:

- `cv`: 分割数（k）。`cv=5`なら5-fold
- `scoring`: 評価指標の指定（後述）

### scoring（代表的なもの）

`scoring`には、タスク（回帰・分類）に応じた評価指標を文字列で指定する。

**回帰タスク向け**

| 値 | 説明 |
| --- | --- |
| `"r2"` | 決定係数（R²）。1に近いほど良い。予測がどれだけ実測値の分散を説明できているかを表す |
| `"neg_mean_squared_error"` | 平均二乗誤差（MSE）の符号を反転したもの。sklearnは「大きいほど良い」スコアに統一する慣習があるため、誤差系の指標は`neg_`が付き符号が反転している。実際のMSEを見るには`-scores`のように符号を戻す |
| `"neg_root_mean_squared_error"` | RMSE（MSEの平方根）の符号反転版。誤差を元のデータと同じ単位で見たいときに使う |
| `"neg_mean_absolute_error"` | 平均絶対誤差（MAE）の符号反転版。外れ値の影響をMSEより受けにくい |

**分類タスク向け**

| 値 | 説明 |
| --- | --- |
| `"accuracy"` | 正解率。全予測のうち正しく分類できた割合 |
| `"precision"` / `"recall"` | 適合率（陽性と予測したうち実際に陽性の割合）／再現率（実際の陽性のうち正しく検出できた割合） |
| `"f1"` | precisionとrecallの調和平均。クラス不均衡データで`accuracy`より信頼できることが多い |
| `"roc_auc"` | ROC曲線の下側面積。二値分類で予測確率の順位付けの良さを評価する |

`scoring`を省略した場合は、モデルの`score()`メソッド（回帰なら`R²`、分類なら`accuracy`）がデフォルトで使われる。

## KMeans

`sklearn.cluster.KMeans` は、データを`k`個のクラスタに分割する代表的なクラスタリング手法。
各クラスタの重心（セントロイド）をランダムに初期化し、「各点を最も近い重心のクラスタに割り当て → 重心を再計算」を
重心が動かなくなるまで繰り返す。正解ラベルを使わずにデータの構造（グループ）を発見する教師なし学習の一種。

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)  # 各サンプルが属するクラスタ番号（0, 1, 2, ...）の配列
```

距離計算を伴うため、PCAと同様にスケールに敏感。特徴量ごとに単位・分散が異なる場合は
`StandardScaler`で標準化してから適用する。

主な引数:

- `n_clusters`: クラスタ数`k`。事前に指定する必要がある（後述のエルボー法・シルエットスコアで選定する）
- `n_init`: 重心の初期化をランダムに変えて実行する回数。KMeansは初期値次第で局所最適に陥ることがあるため、
  複数回実行して最も良い結果（inertiaが最小）を採用する
- `random_state`: 乱数シード。初期化の再現性を固定する

### kの選び方（inertia・silhouette_score）

`k`はハイパーパラメータであり、正解ラベルがないため「これが正解」という基準がない。代わりに以下の指標を使う。

| 指標 | 説明 |
| --- | --- |
| `kmeans.inertia_` | クラスタ内二乗和（各点と所属クラスタ重心との距離の二乗の合計）。小さいほど密なクラスタだが、`k`を増やすほど単調減少するため、減少が緩やかになる「肘」の位置を目視で探す（エルボー法） |
| `sklearn.metrics.silhouette_score` | 各点が自分のクラスタにどれだけ馴染んでいるかを-1〜1で定量化し、全点で平均した値。1に近いほど良い。`k`ごとに算出し、最大値を取る`k`を選べる点がエルボー法と異なり、数値で直接最適な`k`を示せる |

```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):  # k=1ではクラスタが1つしかなく計算できないため2から
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia = kmeans.inertia_
    score = silhouette_score(X, kmeans.labels_)
```

### DBSCANとの違い

`sklearn.cluster.DBSCAN`はクラスタ数を指定せず、密度に基づいてクラスタを発見する。ノイズ点（どのクラスタにも
属さない外れ値）を検出できる点がKMeansと異なるが、高次元データでは密度差が出にくく、`eps`・`min_samples`という
別のハイパーパラメータに敏感になる。
