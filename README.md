# 機械学習勉強

## 汎用

- `docs`: 機械学習・統計の座学中心フォルダ
    - 学習時に気づいたこと、学んだことをドキュメントにしてまとめる
- `scripts`: 機械学習・統計とは直接関係ない、便利スクリプトのフォルダ
    - プロジェクトの準備等を便利にするスクリプトを配置

## 学習

- `handson-ml3`: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow 3rd Edition の学習用フォルダ
    - ノートブックをさらに詳細なテーマに分けて学習していく
    - 学習内容は `handson-ml3/{notebook-number}/README.md` に配置する

## プロジェクト

**実際に学んだ内容を、実データに適用してみる**

- `kaggle`: Kaggle データを用いた学習
- `mlb`: MLBの投手データを用いた分析・予測の学習用フォルダ

## データ配置

- ルート直下も含めて、主要なサブフォルダには、 `README.md` を配置して、サブフォルダの役割を明確にする
    - AI エージェント用に `AGENTS.md`・`CLAUDE.md` も配置する
    - 多くの場合は、 `@README.md` のように `README` と同様でよいが、人間・AIエージェントで知るべき情報の責務が異なるときは分けて書く