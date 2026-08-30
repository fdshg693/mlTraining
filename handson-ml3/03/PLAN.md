# 03_classification 学習プラン

対象: [../converted/03_classification.py](../converted/03_classification.py)（原本: [../original/03_classification.ipynb](../original/03_classification.ipynb)）
学習フォルダ: このフォルダ（`learn` / `exercise` / `experiment` / `mine` / `reinvent` / `data_produce` の構成は[../README.md](../README.md)を参照）

MNIST手書き数字データセットを題材に、分類問題の基本と性能評価指標を学ぶ章。

## フェーズ1: セットアップ & データ準備 (`data_produce`)

- [x] [../converted/03_classification.py:52-70](../converted/03_classification.py#L52-L70) MNISTデータの取得（`fetch_openml`）とデータ構造の確認（`data`/`target`、形状 70000×784）
- [x] [../converted/03_classification.py:74-97](../converted/03_classification.py#L74-L97) 画像の可視化（`plot_digit`）、訓練/テスト分割（先頭60000件が訓練用）
- [x] `data_produce/`にMNIST取得スクリプトを追加し、`README.md`の「未着手」を更新する

## フェーズ2: 二値分類の基礎 (`learn`)

- [x] [../converted/03_classification.py:99-109](../converted/03_classification.py#L99-L109) 「5か否か」の二値分類ラベル作成と`SGDClassifier`での学習・予測
- [x] `random_state`固定の意味、`SGDClassifier`のオンライン学習的な性質を確認する

## フェーズ3: 性能評価指標 (`learn`)

- [x] [../converted/03_classification.py:113-142](../converted/03_classification.py#L113-L142) 交差検証によるAccuracy測定、`DummyClassifier`との比較（Accuracyだけでは不十分な理由の体感）
- [x] [../converted/03_classification.py:146-177](../converted/03_classification.py#L146-L177) 混同行列・適合率(Precision)・再現率(Recall)・F1スコアの算出と手計算での検算
- [x] [../converted/03_classification.py:181-256](../converted/03_classification.py#L181-L256) 決定関数のスコアと閾値、適合率/再現率のトレードオフ、PR曲線の描画
- [x] [../converted/03_classification.py:260-338](../converted/03_classification.py#L260-L338) ROC曲線とAUC、`RandomForestClassifier`との比較（`predict_proba`の利用）
- [x] `exercise`: 別の閾値（例: 目標再現率90%）を設定し、そのときの適合率を求める課題を追加する

## フェーズ4: 多クラス分類 (`learn`)

- [x] [../converted/03_classification.py:344-391](../converted/03_classification.py#L344-L391) `SVC`によるOvO方式、`OneVsRestClassifier`、`SGDClassifier`の多クラス対応、`StandardScaler`によるスケーリング効果
- [x] OvO/OvRの違いと計算量トレードオフを説明できるようにする

## フェーズ5: エラー分析 (`learn` → `experiment`)

- [x] [../converted/03_classification.py:397-470](../converted/03_classification.py#L397-L470) 混同行列の正規化・可視化、誤分類しやすい数字ペア（3と5）の画像確認
- [x] `experiment`: 他の数字ペアで同様のエラー分析を行い、どの数字が混同されやすいか調べる

## フェーズ6: 多ラベル・多出力分類 (`learn`)

- [x] [../converted/03_classification.py:482-509](../converted/03_classification.py#L482-L509) `KNeighborsClassifier`による多ラベル分類（大きい数字/奇数）、`ClassifierChain`
- [x] [../converted/03_classification.py:511-532](../converted/03_classification.py#L511-L532) 多出力分類によるノイズ除去（画像→画像の回帰的分類）

## フェーズ7: 演習 (`exercise`)

- [x] [../converted/03_classification.py:536-569](../converted/03_classification.py#L536-L569) 演習1: `GridSearchCV`で`KNeighborsClassifier`を調整し、テスト精度97%超を達成
- [x] [../converted/03_classification.py:571-640](../converted/03_classification.py#L571-L640) 演習2: `shift()`によるデータ拡張、精度向上とエラー率低下の確認
- [x] [../converted/03_classification.py:642-817](../converted/03_classification.py#L642-L817) 演習3: タイタニックデータセット（前処理パイプライン、`RandomForestClassifier` vs `SVC`の比較、特徴量エンジニアリング）
- [x] [../converted/03_classification.py:819-1121](../converted/03_classification.py#L819-L1121) 演習4: スパム分類器（メール前処理、`EmailToWordCounterTransformer`/`WordCounterToVectorTransformer`の自作、`LogisticRegression`での分類）

## フェーズ8: 自分のデータへの応用 (`mine`)

- [ ] タイタニック・スパム分類器と同様の前処理パイプライン（数値/カテゴリ変数の分離、`ColumnTransformer`）を自分の手持ちデータ（例: MLB投手データ）に適用する

## フェーズ9: 発展 (`reinvent`, 任意)

- [ ] 混同行列・適合率/再現率/F1を`sklearn.metrics`を使わず自前実装し、結果が一致することを確認する
- [ ] 簡易版`WordCounterToVectorTransformer`をscikit-learnの`Pipeline`/`BaseEstimator`規約に沿って自作する

## 完了条件

- [x] 上記フェーズ1〜7を一通り実施し、`README.md`の「学んだ概念」「未学習の概念・今後望まれる学習」を更新する
