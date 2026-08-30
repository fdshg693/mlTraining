# 02_end_to_end_machine_learning_project

## 概要

https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb （Hands-On ML 第2章「エンドツーエンドの機械学習プロジェクト」）に対応する学習フォルダ。

カリフォルニアの住宅価格データを使い、データ取得からモデルの保存・推論まで、回帰プロジェクトを一通り実装した。

- `learn/01_load_and_inspect.py`〜`learn/10_finalize_model.py` — [../converted/02_end_to_end_machine_learning_project.py](../converted/02_end_to_end_machine_learning_project.py)をテーマごとに分割した写経スクリプト（実装済み・実行済み）
- `learn/common.py` — 08以降で使い回す前処理パイプライン（`build_full_preprocessing`）とデータ読込
- `exercise/` — 本文の写経が終わった後の発展課題（SVR比較、特徴量選択、カスタム変換器の完成など）
- [../converted/02_end_to_end_machine_learning_project.py](../converted/02_end_to_end_machine_learning_project.py) — 章のサンプルコード・演習解答をまとめた参照コード（原本は[../original/02_end_to_end_machine_learning_project.ipynb](../original/02_end_to_end_machine_learning_project.ipynb)、変換元スクリプトは[../../scripts/internal/conver_notebook.py](../../scripts/internal/conver_notebook.py)）
- `learn/logs/`・`exercise/logs/` — 各スクリプトの実行ログ（`data_produce/util/logging_config.py`が呼び出し元スクリプトと同じディレクトリに自動生成する）

ここでは、このフォルダを通じて**何の概念を学んだか / 何がまだ学べていないか**をまとめる。

## 学んだ概念

- **代表性を保ったtrain/test分割**
  `learn/02_split_data.py`で単純ランダム分割・ハッシュベースのID分割・`train_test_split`を比較し、`median_income`を5カテゴリに`pd.cut`した上で`StratifiedShuffleSplit`による層化サンプリングがカテゴリ比率のズレを抑えることを確認した。
- **探索的データ分析（EDA）**
  `learn/03_explore_housing.py`で緯度・経度の散布図（人口を大きさ、価格を色に割当）、相関行列、`scatter_matrix`を使い、`rooms_per_house`・`bedrooms_ratio`・`people_per_house`という組み合わせ特徴量を作って元の列との相関を比較した。
- **欠損値・外れ値・カテゴリ値の処理**
  `learn/04_clean_encode.py`で`total_bedrooms`の欠損に対する削除・列削除・中央値補完（`SimpleImputer`）を比較し、`IsolationForest`で外れ値候補を検出、`OrdinalEncoder`と`OneHotEncoder(handle_unknown="ignore")`の違いを確認した。
- **スケーリングと分布変換**
  `learn/05_scale_and_transform.py`で`MinMaxScaler`/`StandardScaler`、長い裾を持つ特徴量への対数変換、`rbf_kernel`による類似度特徴量、`TransformedTargetRegressor`による目的変数の変換・逆変換を確認した。
- **scikit-learn APIに沿ったカスタム変換器**
  `learn/06_custom_transformers.py`で`FunctionTransformer`、および`BaseEstimator`/`TransformerMixin`を継承した`StandardScalerClone`・`ClusterSimilarity`（KMeansクラスタ中心とのRBF類似度）を実装し、`check_is_fitted`・`get_feature_names_out()`の役割を確認した。
- **Pipeline / ColumnTransformerによる前処理の統合**
  `learn/07_preprocessing_pipeline.py`で数値列（中央値補完→標準化）・カテゴリ列（最頻値補完→One-Hot）・比率／対数／クラスタ類似度パイプラインを`ColumnTransformer`にまとめ、`learn/common.py`の`build_full_preprocessing()`として08以降で再利用した。
- **複数モデルの比較と交差検証**
  `learn/08_compare_models.py`で線形回帰・決定木・ランダムフォレストを同じ前処理込みパイプラインで`cross_val_score`比較し、訓練誤差だけではモデルを選べないこと（決定木の過学習）を確認した。
- **ハイパーパラメータ探索**
  `learn/09_tune_model.py`でランダムフォレストに対し`GridSearchCV`と`RandomizedSearchCV`（`randint`/`loguniform`等）を適用し、`cv_results_`で探索結果を比較した。
- **特徴量重要度・最終評価・モデル永続化**
  `learn/10_finalize_model.py`で最良モデルの特徴量重要度を変換後の列名と対応づけ、テストセットに一度だけ適用（テストRMSE 41,549、95%信頼区間 [39,579, 43,805]）し、`joblib`で前処理込みモデルを保存・再読込して予測が一致することを確認した。
- **SVRの追加比較（演習1・2）**
  `exercise/exercise_svr.py`で前処理後にSVRを接続し、線形/RBFカーネルを`GridSearchCV`・`RandomizedSearchCV`（`loguniform`/`expon`）で探索し、ランダムフォレストと比較した（計算量抑制のため訓練データ先頭5,000件・3-fold）。
- **SelectFromModelによる特徴量選択（演習3）**
  `exercise/exercise_feature_selection.py`で`RandomForestRegressor`ベースの`SelectFromModel`を挿入し、`threshold`ごとの選択列数とSVRのCV RMSEを比較、特徴量を減らせば必ず性能が上がるわけではないことを確認した。
- **回帰器の予測を特徴量にするカスタム変換器（演習4・5）**
  `exercise/exercise_regressor_feature.py`で`FeatureFromRegressor`（`MetaEstimatorMixin`付き）を実装し、緯度・経度からのKNN回帰予測を`ClusterSimilarity`と差し替えて性能を比較、その後KNN・SVRのパラメータを同時探索した。
- **StandardScalerCloneの完成（演習6）**
  `exercise/exercise_scaler_clone.py`で`with_mean=False`対応、定数列のゼロ除算防止、`inverse_transform()`、`validate_data`による列名・列数検証を実装し、`check_estimator()`でscikit-learn推定器APIへの適合を確認した。
- **ClusterSimilarityのn_clusters・gammaが空間表現に与える影響（experiment）**
  `experiment/cluster_sweep.py`で`n_clusters`（3, 10, 30, 50、gamma=1.0固定）と`gamma`（1.0, 0.1, 0.05, 0.01、n_clusters=1固定）をそれぞれ振り、地図上の類似度分布として可視化した。`n_clusters`を増やすほど各地区が最寄りクラスタに近くなり最大類似度の平均は0.604→0.971へ上昇・地区間のばらつき（標準偏差）は0.295→0.058へ縮小（＝特徴量は全体的に「効きやすく」なる一方、地区を区別する情報量は下がる）。`gamma`を1.0→0.01へ下げると、類似度0.5以上が「効く」地区の割合が2.9%→100%へ拡大（＝`gamma`が大きいほど中心からの「届く距離」が急速に狭まる）。09の数値最適化（CV RMSEのみ）が裏側でどんな空間パターンに対応しているかを視覚的に補完した。
- **分割方法（単純ランダム vs 層化）とシードによるテストRMSEのブレ（experiment）**
  `experiment/split_strategy_sweep.py`で単純ランダム分割とmedian_income層化分割それぞれについて`random_state`を0〜9まで振り、RandomForestRegressorのテストRMSEを箱ひげ図で比較した。平均テストRMSEはrandom=47,174 / stratified=47,015（差159）、標準偏差はrandom=1,033 / stratified=1,089とほぼ同水準で、箱ひげ図も大きく重なった。01の`split_seed_sweep.py`ではlifesat（27行）で分割方法・シードによる評価指標のブレが顕著だったのに対し、住宅データ（20,640行）ではデータ量が増えるほどシードのブレも層化の効果も相対的に小さくなることを数値で確認した（層化分割がincome_cat比率のズレを抑える効果自体は`learn/02_split_data.py`で確認済みだが、それが最終的なテストRMSEのブレの縮小に直結するとは限らない）。
- **RandomForestのmax_depth・n_estimatorsと過学習の定量的な追跡（experiment）**
  `experiment/rf_overfit_sweep.py`で`max_depth`（2, 5, 10, 20, None、n_estimators=100固定）と`n_estimators`（1, 10, 50, 200、max_depth=None固定）を振り、訓練RMSEと5-fold CV RMSEを重ねて描いた。`max_depth`を2→Noneへ深くすると訓練RMSEは80,618→17,548へ単調に下がり続けたが、CV RMSEは50,389（depth=10）→47,367（depth=20）→47,300（depth=None）とdepth=20あたりでほぼ下げ止まり、そこから先は訓練RMSEだけが下がり続けた（＝訓練誤差とCV誤差の乖離が過学習の進行を可視化する）。`n_estimators`は1→50でCV RMSEが70,074→47,585と大きく改善する一方、50→200の改善はわずか496にとどまり、CV RMSEのブレ（標準偏差）も1,336→452へ縮小した（＝本数を増やすほど個々の木の分散が平均化されるが、改善幅もブレの縮小も逓減する）。08で定性的に確認した「決定木は訓練誤差だけでは判断できない」を、RandomForestの2パラメータについて連続的な曲線として再確認した。
- **上限打ち切りされた目的変数（$500,001）の扱い（experiment）**
  `experiment/capped_value_experiment.py`で、上限付近（$500,000以上、テストの5.0%・205行）の行を(a)そのまま含める、(b)訓練データから削除する、(c)分類器で上限を判定しCAP_VALUE（$500,001）をそのまま出力する（回帰の外挿は諦める）の3パターンで比較した。(b)は非上限部分のテストRMSEを43,837→42,686へわずかに改善させたが、上限部分のRMSEは93,201→116,224へ大幅に悪化した（＝RandomForestは訓練データの値域を超えて外挿できないため、上限行を学習から除くとその価格帯を予測する能力そのものを失う）。(c)は上限判定分類器（precision=0.94, recall=0.57）を挟むことで非上限部分のRMSEを(b)並み（42,697）に保ちつつ、上限部分のRMSEを102,621へ抑えた（(a)ほどではないが(b)より改善）——分類器のrecallが0.57にとどまり見逃した上限行は(b)と同じ外挿の弱さを引きずるため、フラグ化の効果は分類器の検出力に律速されることが分かった。「削除すると全体RMSEが改善する」は非上限部分に限った話で、実運用でのカバレッジ（高額住宅の予測能力）とのトレードオフを伴うことを数値で確認した。
- **不純度ベース重要度とPermutation Importanceの比較（experiment）**
  `experiment/importance_comparison.py`で、`10_finalize_model.py`が保存した最終モデル（n_clusters=45, max_features=9）に対し、変換後の特徴量空間（59列）を揃えた上で`feature_importances_`（MDI）と`permutation_importance`（テストセット、scoring=r2, n_repeats=10）を比較した。事前の予想は「高カーディナリティなOne-Hot列（ocean_proximity）はMDIが過大評価する」だったが、実際はカテゴリ列（One-Hot）合計の重要度はimpurity=8.3%→permutation=9.4%とpermutationの方がむしろ高く、予想は外れた。一方、緯度・経度から作られ互いに強く相関するクラスタ類似度列（45列合計）は impurity=51.7%→permutation=49.8%とグループ合計では小幅な低下にとどまったが、個々のクラスタ列単位では順位が最大±14変動するものが見られた（＝相関する特徴量群は、1列ずつ独立に並べ替えるpermutation_importanceでは他列が情報を代替してしまうため、グループ合計は保たれても個々の列の重要度・順位は不安定になる）。59列中26列で順位が5以上動き、「重要度」は測り方（訓練時の分岐頻度 vs テストでの予測性能への寄与）によって結論が変わり得ることを数値で確認した。

## 未学習の概念・今後望まれる学習

- **データ取得の`data_produce/`への分離**: 現状`learn/01_load_and_inspect.py`が取得を兼ねており、01フォルダのように独立した取得スクリプトへは未分離。
- **Partial Dependenceとの比較**: 重要度の比較（MDI・Permutation）はexperiment/importance_comparison.pyで実施済みだが、特徴量の値と予測の関係（Partial Dependence Plot）はまだ未実施。
- **分布変化・時系列データへの発展**: 今回の分割・評価は分布が変化しないことを前提としており、データ分布が変化する場合の分割・評価は未検証。
- **推論用CLI/APIへの接続**: 保存済みモデル（`learn/outputs/final_model.pkl`）を使う推論の入り口はまだない。
