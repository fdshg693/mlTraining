# Titanic 生存予測

Kaggleのタイタニックコンペデータ（[../data/titanic/](../data/titanic/)）を使い、Kaggleコンペ形式（`test.csv`への予測を`gender_submission.csv`と同じ形式で提出する）で一通り取り組む学習用フォルダ。今後の方針・未着手フェーズは[PLAN.md](PLAN.md)を参照。

## フォルダ構成

- `eda_phase1.py`: フェーズ1（EDA・データ理解）のスクリプト。欠損値確認、`Pclass`/`Sex`/`Age`/`SibSp`/`Parch`/`Fare`/`Embarked`と`Survived`の関係の可視化、`Name`/`Ticket`/`Cabin`からの特徴量抽出の当たりをつける
- `baseline_phase2.py`: フェーズ2（ベースライン構築）のスクリプト。数値（`Age`/`SibSp`/`Parch`/`Fare`: 欠損補完+標準化）・カテゴリ（`Pclass`/`Sex`/`Embarked`: 欠損補完+OneHotEncoding）の前処理パイプラインを組み、`LogisticRegression`/`RandomForestClassifier`/`SVC`を10-fold交差検証で比較。最良モデルで`test.csv`への提出用ファイルを生成する
- `data/`: 各スクリプトが出力する整形済みデータ（`train_eda_features.csv`/`test_eda_features.csv` = 元データに`Title`/`Deck`/`FamilySize`/`IsAlone`/`TicketGroupSize`を追加したもの、`submission_phase2.csv` = フェーズ2の提出用ファイル）
- `output/`: 理解に役立つ図・画像の出力先
- `logs/`: 各スクリプトの実行ログ

`handson-ml3`のような`learn`/`exercise`分割は行わず、フェーズごとに1スクリプト（`eda_phase1.py`・`baseline_phase2.py`、以降も追加予定）を置くフラットな構成とする。本フォルダの目的はKaggleコンペ形式で一通り通すことで、複数の学習観点を分けて反復する`handson-ml3`とは性質が異なるため。

## 実行方法

リポジトリルート（`mlTraining/`）から `uv run` で実行する。

```bash
uv run python kaggle/titanic/eda_phase1.py
uv run python kaggle/titanic/baseline_phase2.py
```

## フェーズ1で分かったこと（EDA）

- 欠損値: `Age`(train 19.9%/test 20.6%)、`Cabin`(train 77.1%/test 78.2%)、`Embarked`(train 2件のみ)、`Fare`(test 1件のみ)。train/testで欠損パターンはほぼ同じ傾向で、testだけの欠損対応（`Fare`の中央値補完など）が別途必要になる
- `Sex`(female 74.2% vs male 18.9%)・`Pclass`(1等63.0%→2等47.3%→3等24.2%)で生存率に大きな差があり、単体でも強い特徴量になりそう
- `SibSp`/`Parch`は0人（単独）より1〜2人の方が生存率が高く、人数が増えすぎる（4人以上）と下がる傾向。単独か家族連れかが効いている可能性がある
- `Age`は幼児（0〜5歳程度）の生存率が高い、`Fare`は生存者の方が平均運賃が高い（生存者48.4 vs 非生存者22.1、`Pclass`と相関している可能性）
- `Name`から抽出した敬称（Title）は性別だけより細かい生存率の差を示す（Mrs 79.2% > Miss 69.8% > Master 57.5% > Rare 44.4% > Mr 15.7%）
- `Cabin`先頭文字（Deck）は欠損（Unknown）の生存率が30.0%と低く、既知のデッキ（特にB/D/E）は74〜76%と高い。ただしUnknown以外は各デッキのサンプル数が少ない（15〜59件）点に注意
- 同一チケット番号での同乗人数（TicketGroupSize）も1人（29.8%）より2〜3人（57〜70%）の方が生存率が高く、`FamilySize`（`SibSp+Parch+1`）と似た傾向だが非血縁の同行者も拾える点で異なる特徴量になりうる

詳細なログ・図は`logs/eda_phase1.log`・`output/`配下（`missing_values.png`・`categorical_survival.png`・`numeric_distributions.png`・`name_ticket_cabin_hints.png`）を参照。

## フェーズ2で分かったこと（ベースライン）

- 数値（`Age`/`SibSp`/`Parch`/`Fare`: 中央値補完+標準化）・カテゴリ（`Pclass`/`Sex`/`Embarked`: 最頻値補完+OneHotEncoding）のみのシンプルな前処理でも、10-fold交差検証Accuracyは`LogisticRegression` 80.2%、`RandomForestClassifier` 81.7%、`SVC` 82.5%（`gamma="auto"`）となり、`SVC`が最良だった
- 3モデルの標準偏差（2〜3pt程度）に対して平均スコアの差（80.2%〜82.5%）は大きくないため、この時点でモデル選択を決め打ちにはせず、フェーズ3の特徴量エンジニアリング後に再比較する
- `test.csv`への予測を`gender_submission.csv`と同じ列構成（`PassengerId`, `Survived`）で出力する型を確立できた（`data/submission_phase2.csv`、生存予測の割合36.6%）。実際にKaggleへ提出するかは未定
- `Name`/`Cabin`/`Ticket`由来の特徴量（`Title`/`Deck`/`FamilySize`/`TicketGroupSize`）やフェーズ1で観察した`Age`欠損補完の見直しは、まだこのベースラインに組み込んでいない

詳細なログ・図は`logs/baseline_phase2.log`・`output/model_comparison.png`を参照。

## 次のフェーズ

フェーズ3（特徴量エンジニアリング）へ進む。詳細は[PLAN.md](PLAN.md)を参照。
