# pandas DataFrame

## `.sort_values`

指定した列の値で、行を**並べ替える**。

```python
importance_df.sort_values("importance", ascending=False)
# importance列の値が大きい順に行を並べ替えたDataFrame
```

| 引数 | 意味 |
| --- | --- |
| `by`（第1引数） | 並べ替え基準にする列名（複数列ならリストで指定可） |
| `ascending` | `True`なら昇順（デフォルト）、`False`なら降順 |

デフォルトでは元のDataFrameを変更せず、並べ替え後の**新しいDataFrame**を返す（`inplace=True`を渡せば元を変更できるが、通常は不要）。

## `.reset_index`

行の**インデックス（行ラベル）**を、`0, 1, 2, ...`の連番に振り直す。

```python
importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
# 並べ替え後、バラバラになったインデックスを0始まりの連番に振り直す
```

| 引数 | 意味 |
| --- | --- |
| `drop` | `True`なら古いインデックスを捨てる。`False`（デフォルト）だと古いインデックスが`index`という新しい列として残る |

### なぜ`sort_values`の後に必要か

`sort_values`は値だけを並べ替え、各行が元々持っていたインデックス（例: 元のDataFrameでの行番号）はそのまま行に付いてくる。
そのため並べ替え後のインデックスは`3, 17, 0, 42, ...`のように飛び飛びになる。

`.loc[0]`で1位の行を取りたい、あるいは単に見た目を整えたいといった場面では、`reset_index(drop=True)`で
インデックスを`0, 1, 2, ...`に振り直しておくと扱いやすい。`drop=True`を付けないと、古いインデックスが
`index`という余分な列として残ってしまう点に注意。

[handson-ml3/02/learn/10_finalize_model.py](../handson-ml3/02/learn/10_finalize_model.py)の
`summarize_feature_importances`では、特徴量重要度を降順に並べ替えた上でインデックスを振り直し、
`importance_df.head(10)`（上位10件）をそのまま「重要度1位、2位、…」として扱えるようにしている。
