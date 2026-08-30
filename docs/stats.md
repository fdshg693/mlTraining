# 統計・評価指標

分類モデルの性能評価に関する統計用語をまとめる。

参照コード: [handson-ml3/03/learn/03_confusion_matrix_metrics.py](../handson-ml3/03/learn/03_confusion_matrix_metrics.py)

## 混同行列（Confusion Matrix）

二値分類の予測結果を「実際のクラス × 予測したクラス」で集計した2×2の表。`sklearn.metrics.confusion_matrix(y_true, y_pred)`で得られ、行が正解ラベル・列が予測ラベルに対応する。

|              | 予測: 陰性 | 予測: 陽性 |
| ------------ | ---------- | ---------- |
| **実際: 陰性** | TN         | FP         |
| **実際: 陽性** | FN         | TP         |

- **TP（True Positive）**: 陽性を陽性と正しく予測
- **TN（True Negative）**: 陰性を陰性と正しく予測
- **FP（False Positive）**: 陰性を誤って陽性と予測（第一種の過誤）
- **FN（False Negative）**: 陽性を誤って陰性と予測（第二種の過誤）

`confusion_matrix`は`cm[0, 0]=TN, cm[0, 1]=FP, cm[1, 0]=FN, cm[1, 1]=TP`の順で返る（クラスラベルの昇順が行・列の並びになるため、`False`が0番目・`True`が1番目に対応する2値分類の場合）。

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
```

## 適合率（Precision）

「陽性と予測したもののうち、実際に陽性だった割合」。誤検知（FP）を避けたい場面で重視する指標。

```
precision = TP / (TP + FP)
```

`sklearn.metrics.precision_score(y_true, y_pred)`で算出できる。

## 再現率（Recall）

「実際に陽性だったもののうち、正しく陽性と検出できた割合」。真陽性率（True Positive Rate）や感度（Sensitivity）とも呼ばれる。見逃し（FN）を避けたい場面で重視する指標。

```
recall = TP / (TP + FN)
```

`sklearn.metrics.recall_score(y_true, y_pred)`で算出できる。

## 適合率と再現率のトレードオフ

一般に適合率と再現率はトレードオフの関係にある。「陽性と判断する基準」を厳しくすると誤検知（FP）が減って適合率は上がるが、見逃し（FN）が増えて再現率は下がる。逆に基準を緩めると再現率は上がるが適合率は下がる。そのため片方だけを見て性能を語ることはできず、両方を併記するか、後述のF1スコアのような統合指標を使う。

## PR曲線とprecision_recall_curveの戻り値

`sklearn.metrics.precision_recall_curve(y_true, y_score)`は、ありうる全ての閾値について適合率・再現率を計算し、`(precisions, recalls, thresholds)`の3つの配列を返す。

- `thresholds`: `y_score`に現れるユニークな値を**昇順**にソートしたもの。長さ`n_thresholds`。
- `precisions`, `recalls`: 長さ`n_thresholds + 1`。**末尾に1要素多い**のは、「どんなスコアでも陽性と予測しない（閾値が無限大）」という、対応する`thresholds`の値が存在しないケースを表すため（`precisions[-1] = 1.0`, `recalls[-1] = 0.0`）。そのため`thresholds`と要素単位で対応させたい場合は`precisions[:-1]`, `recalls[:-1]`を使う（[handson-ml3/03/learn/04_precision_recall_tradeoff.py](../handson-ml3/03/learn/04_precision_recall_tradeoff.py)の`plot_precision_recall_vs_threshold`参照）。

`thresholds[i]`は「スコア >= `thresholds[i]`のとき陽性と予測する」という閾値に対応し、そのときの適合率・再現率が`precisions[i]`, `recalls[i]`になる。

### 閾値探索でargmaxを使うときの前提

`thresholds`が昇順である（＝配列のインデックスが小さいほど閾値も小さい）ことを利用すると、「目標の適合率/再現率を満たす最小/最大の閾値」を`argmax`で探索できる。

```python
# 適合率90%を初めて満たす最小の閾値
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
```

`argmax()`はブール配列に対しては「最大値（=True）が最初に現れるインデックス」を返す。`thresholds`が昇順に並んでいるからこそ、「最初にTrueになるインデックス」＝「適合率90%を達成する最小の閾値のインデックス」と言える（`thresholds`の並び順に関する前提がなければ、この読み替えは成り立たない）。再現率で同様に「目標を満たす最大の閾値」を探す場合は`(recalls >= target).argmin() - 1`のように、再現率が単調非増加であることを使って探す（[exercise/01_recall_90_threshold.py](../handson-ml3/03/exercise/01_recall_90_threshold.py)）。

注意点:

- 再現率は閾値を上げると単調に非増加だが、**適合率は閾値を上げても単調に増加するとは限らない**（局所的に上下することがある）。そのため`argmax`が返すのは「適合率が90%を初めて上回った点」であり、それより大きい閾値で再び90%を下回らない保証はない。
- 条件を満たす要素が1つも無い場合（例: どの閾値でも適合率が90%に届かない）、`(precisions >= 0.90)`は全要素`False`になり、`argmax()`は無条件に`0`を返す（＝全部Falseの配列でも「最初のFalse」が返る）。これは「達成できた」ことを意味しないため、事前に`.any()`で存在確認するのが安全。

## F1スコア（F1 Score）

適合率と再現率の**調和平均**（harmonic mean）。両方がバランス良く高くないと大きな値にならない性質があり、片方だけが極端に高い（例: 再現率100%だが適合率が低い）ケースを低く評価できる。

```
f1 = 2 * (precision * recall) / (precision + recall)
   = TP / (TP + (FN + FP) / 2)
```

単純平均（算術平均）ではなく調和平均を使う理由は、小さい方の値の影響を強く受けるようにするため。例えば適合率100%・再現率0%なら算術平均は50%になってしまうが、調和平均（F1）は0%になり、「使い物にならない」ことを正しく反映する。

`sklearn.metrics.f1_score(y_true, y_pred)`で算出できる。

## クラス不均衡とAccuracyの限界

正解率（Accuracy、全予測中の正答割合）は直感的だが、クラス比が偏ったデータでは性能を見誤らせる。例えば陽性が全体の10%しかないデータで「常に陰性と予測する」だけの分類器でもAccuracyは90%に達してしまう。混同行列・適合率・再現率・F1スコアは、こうしたクラス不均衡下でもTP/FP/FN/TNの内訳から実質的な性能を評価できる。
