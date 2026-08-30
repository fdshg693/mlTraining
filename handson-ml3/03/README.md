# 03_classification

## 概要

https://github.com/ageron/handson-ml3/blob/main/03_classification.ipynb （Hands-On ML 第3章「分類」）に対応する学習フォルダ。

MNIST手書き数字データセットを題材に、二値分類・多クラス分類・多ラベル分類・多出力分類と、それらの性能評価指標（混同行列、適合率・再現率、PR曲線・ROC曲線など）を扱う章。

フォルダ構成の全体像・各サブフォルダの役割は[../README.md](../README.md)を参照。現時点では`data_produce/`にMNIST取得スクリプト、`learn/`に二値分類〜多ラベル・多出力分類のスクリプト、`exercise/`に閾値探索・章末演習(KNNチューニング、データ拡張、タイタニック、スパム分類器)、`experiment/`にエラー分析を広げた数字ペア探索を追加済みで、`mine`・`reinvent`はこれから追加する。

- `data_produce/` — MNISTデータの取得・前処理を行う層（[data_produce/fetch_mnist.py](data_produce/fetch_mnist.py)で取得し、訓練60000件/テスト10000件に分割して`data/`へ保存する。原本は`fetch_openml('mnist_784')`を使うが、OpenMLのAPIが504エラーで取得できなかったため、`tensorflow.keras.datasets.mnist`が内部で使う公開npz（storage.googleapis.com）から直接取得する形に変更している）
- `learn/01_binary_classifier.py` — 「5か否か」の二値分類ラベル作成、`SGDClassifier`での学習・予測、`random_state`固定の意味、`partial_fit`によるオンライン学習的な性質の確認（実装済み・実行済み）
- `learn/02_cross_validation_accuracy.py` — `cross_val_score`によるAccuracy測定、`StratifiedKFold`での手動再現、`DummyClassifier`との比較（実装済み・実行済み）
- `learn/03_confusion_matrix_metrics.py` — 混同行列・適合率・再現率・F1スコアの算出と、混同行列からの手計算による検算（実装済み・実行済み）
- `learn/04_precision_recall_tradeoff.py` — 決定関数のスコアと閾値、適合率/再現率のトレードオフ、PR曲線の描画（実装済み・実行済み）
- `learn/05_roc_curve.py` — ROC曲線とAUC、`RandomForestClassifier`（`predict_proba`）との比較（実装済み・実行済み）
- `exercise/01_recall_90_threshold.py` — 目標再現率90%を満たす閾値とそのときの適合率を求める演習（実装済み・実行済み）
- `learn/06_multiclass_classification.py` — `SVC`によるOvO方式、`OneVsRestClassifier`によるOvR方式、`SGDClassifier`の多クラス対応、`StandardScaler`によるスケーリング効果（実装済み・実行済み）
- `learn/07_error_analysis.py` — 混同行列の正規化・可視化（行正規化、誤分類のみに絞った行/列正規化）、誤分類しやすい数字ペア（3と5）の画像確認（実装済み・実行済み）
- `experiment/01_confused_digit_pairs.py` — `learn/07`の3と5のペア分析を全数字ペアに拡張し、誤分類件数が多い上位5ペアをランキング・画像確認する実験（実装済み・実行済み）
- `learn/08_multilabel_multioutput.py` — `KNeighborsClassifier`による多ラベル分類（「7以上か」「奇数か」）と`cross_val_predict`によるF1スコア評価、`ClassifierChain`による多ラベル分類、`KNeighborsClassifier`による多出力分類（ノイズ除去、画像→画像の回帰的分類）（実装済み・実行済み）
- `exercise/02_knn_grid_search_97.py` — 演習1: `GridSearchCV`で`KNeighborsClassifier`の`weights`/`n_neighbors`を先頭10000件でチューニングし、最良パラメータを訓練データ全体で再学習してテスト精度97%超を達成（実装済み・実行済み。最良パラメータとテスト精度は`exercise/outputs/best_knn_params.json`に保存し、演習2で再利用）
- `exercise/03_data_augmentation.py` — 演習2: `scipy.ndimage.shift()`で上下左右に1画素シフトした画像を追加し訓練データを5倍（60000→300000件）に拡張、演習1の最良パラメータで再学習してテスト精度向上とエラー率低下を確認（実装済み・実行済み）
- `exercise/04_titanic.py` — 演習3: タイタニックデータセットを`ColumnTransformer`で数値/カテゴリ変数に分けて前処理し、`RandomForestClassifier`と`SVC`を10-fold交差検証で比較、年齢層・同乗家族数による特徴量エンジニアリングの手がかりを確認（実装済み・実行済み）
- `exercise/05_spam_classifier.py` — 演習4: Apache SpamAssassinの公開コーパス（ham/spam）を取得し、`EmailToWordCounterTransformer`（HTML除去・小文字化・数字/URL置換・stemming）と`WordCounterToVectorTransformer`（語彙選定・スパースベクトル化）を自作、`LogisticRegression`でスパム分類（実装済み・実行済み）
- [../converted/03_classification.py](../converted/03_classification.py) — 章のサンプルコード・演習解答をまとめた参照コード（原本は[../original/03_classification.ipynb](../original/03_classification.ipynb)、変換元スクリプトは[../../scripts/internal/conver_notebook.py](../../scripts/internal/conver_notebook.py)）

## 学んだ概念

- **MNISTのデータ構造**
  70000枚×784画素（28×28グレースケール）の画像と、文字列としての数字ラベルという構造を`data_produce/fetch_mnist.py`で確認した。原本の`fetch_openml('mnist_784')`は(N, 784)の平坦化済み配列を返すが、代替取得元の`tensorflow.keras.datasets.mnist`用npzは(N, 28, 28)のuint8画像として提供されるため、`reshape(-1, 784)`で形を揃える必要があった。
- **訓練/テスト分割の慣習**
  MNISTは先頭60000件が訓練用・残り10000件がテスト用という前提を持つ。代替取得元は最初からこの分割で提供されているため、原本のような手動スライスは不要だった。
- **外部データ取得の可用性リスク**
  OpenMLのAPIが504 Gateway Timeoutで一時的に利用不能だったため、同じMNISTを提供する別のホスティング元（Keras用npz配布）に切り替えた。データ取得層を薄く抽象化しておくことで、取得元の差し替えが後続の`learn`層に影響しないようにした。
- **画像データの可視化によるサニティチェック**
  784次元ベクトルを28×28に`reshape`して`imshow`することで、取得したデータが正しい画像・ラベル対応になっているかを目視確認できる。
- **二値分類ラベルの作成とSGDClassifierによる学習・予測**
  `learn/01_binary_classifier.py`で`y_train == '5'`により「5か否か」の二値ラベルを作り、`SGDClassifier(random_state=42)`で学習・予測した。`X_train[0]`（ラベル'5'）に対する予測は`True`、テスト精度は94.92%だった。
- **random_state固定の意味**
  同じ`random_state=42`で2回学習すると決定関数の出力は完全に一致（差0）する一方、`random_state=0`に変えると決定関数が大きくずれ（絶対値最大差 約11,654）、`random_state`未指定（実行のたびに異なるシード）でも同様にずれる（約4,897）ことを確認した。`SGDClassifier`は内部で訓練データをシャッフルしながら勾配降下するため、シードの違いがサンプルの処理順・ひいては学習される重みの違いに直結する。
- **SGDClassifierのオンライン学習的な性質**
  `fit`による一括学習（テスト精度94.92%）と、`partial_fit`を用いてbatch_size=1000のミニバッチに分けて逐次学習した場合（テスト精度96.10%）を比較した。`partial_fit`は訓練データ全体を一度にメモリへ載せず、ミニバッチ単位で重みを逐次更新できるため、大規模データやストリーミングデータに向く。
- **Accuracyだけでは不十分な理由（クラス不均衡）**
  `learn/02_cross_validation_accuracy.py`で、常に「5でない」と予測するだけの`DummyClassifier`の交差検証Accuracyが約90.97%（訓練データ中「5でない」割合とほぼ一致）になることを確認した。`SGDClassifier`のAccuracy（約95〜96%）と大差ないため、クラス比が偏ったデータではAccuracy単体では性能を評価できない。また`cross_val_score`は内部で`StratifiedKFold`と同等の分割・評価を行っており、手動再現したfold accuracyと完全一致（差0）した。
- **混同行列・適合率・再現率・F1スコア**
  `learn/03_confusion_matrix_metrics.py`で、混同行列（TN=53892, FP=687, FN=1891, TP=3530）からprecision_score（0.8371）・recall_score（0.6512）・f1_score（0.7325）を算出し、`TP/(FP+TP)`・`TP/(FN+TP)`・`TP/(TP+(FN+FP)/2)`による手計算と完全一致することを確認した。F1スコアは適合率と再現率の調和平均で、両方が高くないと高い値にならない。
- **決定関数のスコアと閾値、適合率/再現率のトレードオフ**
  `learn/04_precision_recall_tradeoff.py`で、`decision_function`が返すスコアと閾値の比較（`predict()`は閾値0に相当）で予測が決まることを確認した。閾値を上げるほど適合率は上がり再現率は下がるトレードオフがあり、適合率90%を達成する閾値（約3370）ではそのときの再現率は48%まで下がった。
- **ROC曲線とAUC、RandomForestClassifierとの比較**
  `learn/05_roc_curve.py`で、`SGDClassifier`のROC AUC（0.9605）と、`predict_proba`を利用した`RandomForestClassifier`のROC AUC（0.9983）を比較し、確率出力を使えるRandomForestの方が上回ることを確認した。また陽性確率50〜60%と予測された画像のうち実際に陽性だった割合が94.0%となり、`predict_proba`の出力が実際の確率としてよく較正されていることも確認した。
- **目標再現率からの閾値探索（演習）**
  `exercise/01_recall_90_threshold.py`で、`learn/04`とは逆に目標再現率90%を満たす最大の閾値（約-6862）を探索し、そのときの適合率（0.5161）を求めた。同じ適合率/再現率トレードオフ曲線上でも、どちらを目標にするかで得られる閾値・もう一方の指標の値が大きく変わることを確認した。
- **SVCによるOvO方式の多クラス分類**
  `learn/06_multiclass_classification.py`で、`SVC`は内部で常にOvO（One-vs-One、クラスペアごとに分類器を学習し多数決）方式を使うことを確認した。`decision_function`は既定でOvRに集約された10個のスコアを返すが、`decision_function_shape="ovo"`にすると10クラス×9/2=45個のペアワイズスコアがそのまま得られる（学習方式自体は変わらず、スコアの見せ方だけが変わる）。先頭2000件のみで学習しても`X_train[0]`（ラベル'5'）は正しく'5'と予測できた。
- **OneVsRestClassifierによる明示的なOvR方式**
  `OneVsRestClassifier(SVC())`で明示的にOvR（One-vs-Rest、各クラスvs残り全部の2値分類器を学習）方式を試し、内部に10個（クラス数分）の2値分類器が保持されることを確認した。
- **OvO/OvRの計算量トレードオフ**
  OvOはクラスペア数（45個）の分類器を各ペアのデータのみで学習し、OvRはクラス数（10個）の分類器をそれぞれ全訓練データで学習する。分類器1つあたりの訓練データが少ないOvOと、分類器数が少ないOvRという計算量のトレードオフがあり、訓練データが増えるほど遅くなるSVMのようなアルゴリズムではOvOが好まれる。
- **SGDClassifierの多クラス対応とStandardScalerの効果**
  `SGDClassifier`はクラスラベルをそのまま渡すだけで多クラス分類（内部でOvR）に対応する。60000件全体での3-fold交差検証Accuracyはスケーリング前が平均86.85%、`StandardScaler`で画素値を平均0・分散1に標準化した後は平均89.65%に向上した。勾配降下ベースの`SGDClassifier`は特徴量のスケールが揃っている方が収束しやすいことを確認した（なお`max_iter`未到達の`ConvergenceWarning`が出ており、収束自体は不完全）。

- **混同行列の正規化による可視化改善**
  `learn/07_error_analysis.py`で、`StandardScaler`後のデータに対する`SGDClassifier`の3-fold交差検証予測から混同行列を作成した。件数そのままの混同行列は正解数の多いクラス（対角成分: 4445〜6398件）が目立ちやすく見た目での比較が難しいが、`normalize="true"`で行（真のクラス）ごとに正規化すると、各クラス内での予測先の割合として比較しやすくなる。さらに`sample_weight`で正解（対角成分）の重みを0にし、誤分類のみに絞って行/列それぞれで正規化すると、どの数字がどの数字に間違われやすいかがより明確になった。
- **誤分類しやすい数字ペアの画像確認**
  `learn/07_error_analysis.py`で、'3'と'5'の混同行列上のセルに対応する画像を4象限（'3'正解5224件・'5'正解4445件、'3'→'5'誤分類204件・'5'→'3'誤分類165件）のグリッドとして可視化した。誤分類された画像は曲線の丸みが似ており、人間の目にも紛らわしい筆跡が多いことを確認した。エラー分析は精度指標だけでなく、モデルがどこで「人間的にも納得できる」間違え方をしているかを確認する手段になる。
- **全数字ペアでの誤分類ランキング（実験）**
  `experiment/01_confused_digit_pairs.py`で、全45クラスペアについて誤分類件数（a→b + b→a）を集計しランキングした。上位5ペアは`'5'<->'8'`（665件）、`'3'<->'8'`（494件）、`'8'<->'9'`（432件）、`'2'<->'8'`（427件）、`'7'<->'9'`（391件）で、実行前に予想していた`'4'<->'9'`は上位に入らず、代わりに数字'8'が上位5ペア中4つに登場した。誤分類の正規化混同行列で'8'の行・列に高い割合が集中していたのと一致しており、'8'は丸みのある形が他の複数の数字（2, 3, 5, 9）と部分的に重なりやすく、モデルにとって最も紛らわしいクラスになっていることが分かった。

- **KNeighborsClassifierによる多ラベル分類**
  `learn/08_multilabel_multioutput.py`で、「7以上か」「奇数か」の2つの二値ラベルを列として持つ`y_multilabel`配列（`np.c_`で結合）を作り、`KNeighborsClassifier`にそのまま渡すだけで多ラベル分類ができることを確認した。`X_train[0]`（ラベル'5'）に対する予測は`[False, True]`（7以上でない・奇数）で正解と一致した。3-fold交差検証による予測から算出したF1スコアは`average="macro"`で0.9764、`average="weighted"`で0.9778となり、2つのラベルの陽性件数比がほぼ均等なため両者の差はごくわずかだった。
- **ClassifierChainによる多ラベル分類**
  `ClassifierChain(SVC(), cv=3, random_state=42)`を先頭2000件で学習し、`X_train[0]`に対する予測は`[0, 1]`（KNNと同じく7以上でない・奇数）となった。`ClassifierChain`は各ラベルを2値分類器の連鎖として順に予測し、後段の分類器は前段で予測されたラベルも特徴量として利用できる点が、ラベル間の依存関係を考慮しない`KNeighborsClassifier`との違いになる。
- **多出力分類によるノイズ除去**
  `learn/08_multilabel_multioutput.py`で、訓練・テスト画像それぞれに0〜99のランダムノイズ（`np.random.default_rng(42).integers`）を加えた`X_train_mod`/`X_test_mod`を作り、ノイズなしの元画像`y_train_mod=X_train`を教師信号として`KNeighborsClassifier`を学習させた。これは784画素それぞれを0〜255の256クラス分類として同時に予測する「多出力分類」で、ノイズ画像から元の'7'の形状をおおむね復元できることを`noise_removal_plot.png`（ノイズ入力・KNN予測・正解の3枚並び）で確認した。復元画像と元画像の画素値の平均絶対誤差は8.27だった。

- **GridSearchCVによるハイパーパラメータ探索とデータ量の影響**
  `exercise/02_knn_grid_search_97.py`で、デフォルトの`KNeighborsClassifier`（テスト精度96.88%）に対し、先頭10000件のみで`weights`（uniform/distance）×`n_neighbors`（3〜6）の`GridSearchCV`（5-fold）を行うと`best_params_={'n_neighbors': 4, 'weights': 'distance'}`、`best_score_=0.9442`だった。10000件のみの学習ではスコアがベースラインより下がる（データ量が少ないため）が、この最良パラメータを訓練データ全体（60000件）で再学習するとテスト精度97.14%となり、目標の97%超を達成した。少量データでのグリッドサーチで探索コストを抑えつつ、最終的な学習は全データで行うという実務的な手順を確認した。
- **shift()によるデータ拡張(Data Augmentation)**
  `exercise/03_data_augmentation.py`で、`scipy.ndimage.shift()`を使い各訓練画像を上下左右に1画素ずつシフトした複製を作成し、訓練データを60000件から300000件（5倍）に拡張した。演習1の最良パラメータ（`n_neighbors=4, weights="distance"`）で拡張データを学習すると、テスト精度は97.14%→97.63%に向上し、エラー率（1-精度）は17%低下した。精度の伸び自体は小さく見えても、エラー率で見ると下がり幅が大きいことを確認し、ラベルを保った変換によって学習データを人工的に増やす「データ拡張」の効果を体感した。
- **タイタニックデータセットの前処理パイプラインと分類器比較**
  `exercise/04_titanic.py`で、数値属性（Age, SibSp, Parch, Fare）は`SimpleImputer(median)`→`StandardScaler`、カテゴリ属性（Pclass, Sex, Embarked）は`OrdinalEncoder`→`SimpleImputer(most_frequent)`→`OneHotEncoder`という2系統の`Pipeline`を`ColumnTransformer`で束ねる前処理を組んだ。Age・Cabin・Embarkedに欠損があり（特にCabinは77%欠損のため今回は不使用）、生存率が38.4%とほぼ4割弱のためAccuracyでも評価に使えることを確認した。10-fold交差検証で`RandomForestClassifier`（平均81.4%）と`SVC(gamma="auto")`（平均82.5%）を比較し、SVCの方が平均スコアが高い結果を得た。さらに年齢層（AgeBucket）・同乗家族数（RelativesOnboard=SibSp+Parch）を集計すると`Survived`との相関が見られ、特徴量エンジニアリングの余地があることも確認した。
- **スパム分類器の自作前処理パイプライン**
  `exercise/05_spam_classifier.py`で、Apache SpamAssassinの公開コーパス（ham 2500件・spam 500件）を取得し、`email`モジュールでメールを解析した。ham は`text/plain`が大半（2408/2500）、spam は`text/html`を多く含む（183+multipart系）という構造の違いを確認し、メール構造自体が有用な特徴になりうることを見た。`BaseEstimator`/`TransformerMixin`を継承した`EmailToWordCounterTransformer`（HTML除去・小文字化・数字→NUMBER・URL→URL置換・`nltk.PorterStemmer`による語幹抽出）と`WordCounterToVectorTransformer`（頻出上位1000語を語彙とし`scipy.sparse.csr_matrix`でベクトル化）を自作し、`Pipeline`でつないだ。`LogisticRegression`による3-fold交差検証Accuracyは98.50%、テストデータでの適合率96.88%・再現率97.89%を達成し、自作の前処理パイプラインでも高い適合率・再現率が両立できることを確認した。

## 未学習の概念・今後望まれる学習

- 自分の手持ちデータ（MLB投手データなど）への前処理パイプライン適用は`mine`フォルダで今後実施する
- 混同行列・適合率/再現率/F1・`WordCounterToVectorTransformer`相当の自前実装は`reinvent`フォルダで今後実施する（任意）
