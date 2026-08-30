# SGDClassifier

`sklearn.linear_model.SGDClassifier` は、確率的勾配降下法（Stochastic Gradient Descent）で学習する線形分類器。
1サンプル（またはミニバッチ）ごとに損失関数の勾配を計算して重みを少しずつ更新していくため、
全データを一度に処理する`LogisticRegression`や`SVC`と違い、大規模データやストリーミングデータに強い。

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
prediction = sgd_clf.predict([some_sample])
```

## 損失関数（`loss`）で分類器の種類が変わる

`SGDClassifier`自体は「SGDで学習する」という最適化アルゴリズムの器であり、`loss`引数に何を渡すかで
実質的にどの分類器を近似しているかが決まる。

| `loss` | 近似する分類器 | 特徴 |
| --- | --- | --- |
| `"hinge"`（既定） | 線形SVM | マージン最大化。`decision_function`はマージンからの距離であり、確率ではない |
| `"log_loss"` | ロジスティック回帰 | `predict_proba`が使え、出力を確率として解釈できる |
| `"modified_huber"` | 上記の中間 | 外れ値に頑健で`predict_proba`も使える |

つまり「SVMやロジスティック回帰を、大規模データでも学習できるようにSGDで解く」ためのクラス、という理解がしやすい。

## `decision_function`とは何か

`decision_function`は、サンプルが分類境界からどれだけ離れているかを表す生のスコア（実数値）を返すメソッド。
`predict`は内部でこのスコアを計算し、0以上ならTrue（陽性）、0未満ならFalse（陰性）を返しているだけの薄いラッパーであり、
`decision_function`はその判定の「根拠」を直接見るための手段になる。

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

scores = sgd_clf.decision_function([some_digit])  # 例: array([2164.22])
prediction = scores > 0  # predict()と同じ判定ロジック
```

スコアの意味は`loss`引数によって変わる。既定の`loss="hinge"`（線形SVM相当）では、スコアは分類境界（マージン）からの
符号付き距離であり、確率ではない。値が大きい（絶対値が大きい）ほど「その予測に自信がある」とは言えるが、
「80%の確率で5である」のような確率としては解釈できない。確率として解釈したい場合は`loss="log_loss"`や
`"modified_huber"`にして`predict_proba`を使う必要がある（[損失関数（`loss`）で分類器の種類が変わる](#損失関数loss で分類器の種類が変わる)を参照）。

`decision_function`が特に役立つのは以下のような場面。

- **閾値を0以外に変えたい場合**: `predict`は常に閾値0で判定するが、`decision_function`のスコアに対して
  任意の閾値を適用すれば、適合率・再現率のトレードオフを調整できる
  （`handson-ml3/03/learn/04_precision_recall_tradeoff.py`）。
- **学習された重み自体を比較したい場合**: 同じ入力に対する`decision_function`の出力を2つのモデル間で比較すると、
  `predict`の0/1判定だけでは見えない「学習された重みそのものの違い」を直接確認できる。
  `handson-ml3/03/learn/01_binary_classifier.py`の`explore_random_state`では、この性質を使って
  `random_state`の違いが決定関数の出力（＝学習された重み）にどれだけ影響するかを確認している。
- **ROC曲線・PR曲線を描く場合**: `predict`の0/1判定ではなく、`decision_function`が返す連続値のスコアを
  様々な閾値でスキャンすることで曲線を描ける（`handson-ml3/03/learn/05_roc_curve.py`）。

## オンライン学習的な性質（`fit` vs `partial_fit`）

`fit`はデータ全体を一度に読み込んで学習するのに対し、`partial_fit`はミニバッチ単位で重みを逐次更新できる。
データ全体をメモリに載せる必要がないため、大規模データやストリーミングデータ（後から追加で届くデータ）に向く。

```python
classes = np.array([False, True])  # 初回呼び出し時に全クラスを明示する必要がある
sgd_clf = SGDClassifier(random_state=42)
for start in range(0, len(X_train), batch_size):
    end = start + batch_size
    sgd_clf.partial_fit(X_train[start:end], y_train[start:end], classes=classes)
```

`handson-ml3/03/learn/01_binary_classifier.py`で実際に比較したところ、MNISTの「5か否か」二値分類において
`fit`によるテスト精度94.92%に対し、`batch_size=1000`の`partial_fit`では96.10%と、逐次学習でも遜色ない（むしろ上回る）
精度が得られた。

## `random_state`固定が特に重要な理由

`SGDClassifier`は学習の各エポックで訓練データをシャッフルしながら勾配降下するため、乱数シードが違うと
サンプルの処理順が変わり、収束する重みも変わる。同じ`random_state=42`で2回学習すれば決定関数の出力は
完全に一致（差0）するが、シードを変えたり指定しなかったりすると、実行のたびに結果がぶれる。
再現性が必要な検証・比較では`random_state`を必ず固定する。

## 特徴量のスケーリングに敏感

勾配降下ベースのアルゴリズムであるため、特徴量のスケールが揃っていないと収束が遅くなる。
MNIST（画素値0〜255）を`StandardScaler`で平均0・分散1に標準化すると、3-fold交差検証のAccuracyが
86.85%→89.65%に向上した（`handson-ml3/03/learn/06_multiclass_classification.py`）。
`SGDClassifier`を使う際は`StandardScaler`とセットで使うのが基本。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
```

## 多クラス分類への対応（内部でOvR）

クラスラベルをそのまま渡すだけで多クラス分類に対応する。内部的には
One-vs-Rest（各クラス vs 残り全部の2値分類器をクラス数分学習し、`decision_function`が最大のクラスを選ぶ）方式を使う。
`SVC`が常にOne-vs-One方式を使うのとは対照的（クラス数が多いほど、訓練データが増えるほど学習が遅くなる
アルゴリズムではOvOが好まれ、そうでなければOvRが好まれる傾向がある）。

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)  # y_trainは文字列ラベルでもそのままでよい
scores = sgd_clf.decision_function([some_digit])  # クラス数分のスコアが返る
```

## 主な引数

| 引数 | 説明 |
| --- | --- |
| `loss` | 損失関数（前述）。既定は`"hinge"`（線形SVM相当） |
| `penalty` | 正則化の種類。`"l2"`（既定）、`"l1"`、`"elasticnet"` |
| `alpha` | 正則化の強さ。大きいほど正則化が強い（＝過学習を抑えるが学習不足のリスクも上がる） |
| `max_iter` | 最大エポック数。収束前に打ち切られると`ConvergenceWarning`が出る |
| `random_state` | 乱数シード。シャッフル順を固定し再現性を確保する（前述） |
| `learning_rate` / `eta0` | 学習率のスケジュールと初期値。既定の`"optimal"`は`alpha`から自動計算される |

## LogisticRegression・SVCとの使い分け

- データ量が少なく、正確な最適解が欲しい場合は`LogisticRegression`や`SVC`（バッチ学習・厳密な最適化）を選ぶ
- データ量が非常に大きい、またはデータが逐次到着する（全部を一度にメモリに載せられない）場合は
  `SGDClassifier`を選ぶ
- `SGDClassifier`は近似的な最適化（確率的勾配降下）のため、同じ`loss`でも`LogisticRegression`/`SVC`と
  厳密に同じ境界にはならない点に注意
