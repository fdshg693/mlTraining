# スタッツカタログ取得（`fetch_stat_catalog.py`）

baseballsavantで選択可能な投手・打者スタッツの一覧（`selections`パラメータに指定できる値）を、CSVではなくHTMLのカラム選択UIから取得してYAMLに保存する `data_produce/fetch_stat_catalog.py` の機能概要と検証手順。実装本体は [`mlb/data_produce/CLAUDE.md`](../data_produce/CLAUDE.md) を参照。

## 機能概要

### 目的

`data_produce/savant_client.SavantField` Enumに新しいフィールドを追記する際、「そもそもsavant側にどんなフィールドが存在するか」を毎回ブラウザで確認するのは非効率。[PLAN.md](../PLAN.md)の題材案3・4（クラスタリング・多変量回帰）でも「取得可能なsavantフィールド一覧を調査する必要がある」と指摘されており、その調査を自動化する下調べ用スクリプト。

### データソース

以下のURL（`type=pitcher`または`type=batter`のカスタムリーダーボード画面）のHTMLを取得する。CSVエンドポイントではなく、カラム選択用のチェックボックスUIを持つ通常のHTMLページ。`--type`（`stat_catalog.fetch_stat_catalog`の`player_type`引数）で切り替えるだけで、以下はどちらも同じ実装（`_extract_category`が見出しの祖先`div.flex`配下のチェックボックスを拾う方式）で対応できている。

```
# 投手
https://baseballsavant.mlb.com/leaderboard/custom?year={year}&type=pitcher&filter=&min=q&selections=&chart=false&x=&y=&r=no&chartType=beeswarm&sort=xwoba&sortDir=asc

# 打者
https://baseballsavant.mlb.com/leaderboard/custom?year={year}&type=batter&filter=&min=q&selections=&chart=false&x=&y=&r=no&chartType=beeswarm&sort=xwoba&sortDir=desc
```

### 抽出方法

1. `<h6>Standard Stats</h6>` を探し、その祖先の `<div class="flex">` を見つける
2. そのdiv配下にある `<input type="checkbox" id="...">` をすべて集め、`id`（=savantの`selections`に指定する値）とラベル文字列を対にする
3. `<h6>Statcast Stats</h6>` についても同様に行う

素朴な文字列処理（正規表現で「見出しから次の見出しまで」を抜き出す等）では対応できない罠が2つあったため、**BeautifulSoup（`html.parser`）でDOMツリーとして解析する**方式にした。

- **HTMLコメントに旧定義が残っている**: `Standard Stats`の`div.flex`冒頭には、`<!-- <input id="hit">... -->` のように無効化された古いチェックボックス群がコメントとして残っており、かつ同じ`id`（`hit`/`strikeout`/`walk`等）が下に生きた要素としても存在する。正規表現でテキストとして拾うとコメント内・外の両方がヒットして重複IDになるが、BeautifulSoupはコメントを`Comment`ノードとして扱いタグ検索から除外するため、生きているチェックボックスだけが残る
- **`<label for=...>`とチェックボックスの`id`が一致しない箇所がある**: `arm_angle`/`n_cu_formatted`/`n_si_formatted`など数件で、`<label for="n_arm_angle">`のように`for`属性がtypoでズレている（savant側の実装バグ）。`for`属性で対応する`<label>`を探すとラベルが引けないため、DOM上でチェックボックスの直後にある`<label>`（`find_next_sibling("label")`）を採用する

### 出力

`data/stat_catalog/{player_type}/{year}/stat_catalog.yaml`（+ 同名の`.meta.json`）。`player_type`は`pitcher`/`batter`のどちらでも同じ形式で出力される。

```yaml
# pitcher/2026/stat_catalog.yaml
year: 2026
player_type: pitcher
fetched_at: '2026-08-08T14:47:37+00:00'
url: https://baseballsavant.mlb.com/leaderboard/custom?...
categories:
  standard_stats:
    - id: p_game
      label: G
    - id: p_formatted_ip
      label: IP
    # ...
  statcast_stats:
    - id: ff_avg_speed
      label: 4-Seam Avg MPH
    # ...
```

```yaml
# batter/2026/stat_catalog.yaml
year: 2026
player_type: batter
fetched_at: '2026-08-08T15:51:35+00:00'
url: https://baseballsavant.mlb.com/leaderboard/custom?...
categories:
  standard_stats:
    - id: b_game
      label: G
    - id: batting_avg
      label: AVG
    # ...
  statcast_stats:
    - id: exit_velocity_avg
      label: Avg EV (MPH)
    - id: barrel_batted_rate
      label: Barrel%
    # ...
```

`.meta.json`には取得URL・パラメータ・取得日時を記録する（`fetch_leaderboard.py`のキャッシュ設計を踏襲）。

### 注意点

- `statcast_stats`の親`div.flex`は、UI上「Statcast Stats」だけでなく複数の下位見出しとも箱を共有している。そのためYAMLの`statcast_stats`には、これら全ての見出し配下のスタッツがまとめて含まれる。下位見出しは`player_type`によって異なる（例: 投手ページは「Bat Tracking」「Quality of Contact」「Pitches & Location」「Pitch Arsenals」、打者ページは「Bat Tracking」「Quality of Contact」「Pitches & Location」「Pop Time」「Catch Probability」「Jump」「Sprint Speed」）。カテゴリを見出し単位でさらに細分化したい場合は、`_extract_category`を「次のh6が出るまで」で区切るよう拡張する必要がある（現状は未実装）
- CSV取得系（`fetch_leaderboard.py`）とは別のデータソース（HTMLカラム選択UI）を叩いているため、`stat_catalog.yaml`に載っている`id`が必ず`fetch_leaderboard.py`で実際にCSVカラムとして返ってくる保証はない（未検証）。あくまで「候補一覧」として使う
- `pitcher`/`batter`のどちらも`CATEGORY_HEADINGS`（見出しテキスト→YAMLキーの対応表）・抽出ロジックは共通。`player_type`ごとの分岐はURLの`type`パラメータと保存先パス（`data/stat_catalog/{player_type}/{year}/`）だけで、コード変更なしに両対応している

## 検証手順

1. 依存関係の確認: `beautifulsoup4`・`pyyaml`が`pyproject.toml`に追加されていること
   ```bash
   uv sync
   ```
2. 実行してYAMLが生成されることを確認（`pitcher`/`batter`両方）
   ```bash
   uv run python mlb/data_produce/fetch_stat_catalog.py --year 2026 --type pitcher
   uv run python mlb/data_produce/fetch_stat_catalog.py --year 2026 --type batter
   ```
   - それぞれログに`[standard_stats] N件: [...]`・`[statcast_stats] N件: [...]`が出力されること
   - `mlb/data_produce/data/stat_catalog/pitcher/2026/stat_catalog.yaml`・`mlb/data_produce/data/stat_catalog/batter/2026/stat_catalog.yaml`が生成されること
3. 既知のID（`savant_client.SavantField`に既に登録済みのもの、または実際にsavant UIで確認できるもの）がYAMLに含まれることを目視確認
   - pitcher / `standard_stats`: `p_game`（G）、`p_formatted_ip`（IP）
   - pitcher / `statcast_stats`: `ff_avg_speed`（4-Seam Avg MPH）、`ff_avg_spin`
   - batter / `standard_stats`: `b_game`（G）、`batting_avg`（AVG）
   - batter / `statcast_stats`: `exit_velocity_avg`（Avg EV (MPH)）、`barrel_batted_rate`（Barrel%）
4. 重複がないことを確認（HTMLコメント内の旧定義を誤って拾っていないかのチェック）。`player_type`を`pitcher`/`batter`に変えて両方確認する
   ```bash
   PYTHONPATH=mlb uv run python -c "
   from data_produce.util.file_io import read_yaml
   content = read_yaml('stat_catalog/batter/2026/stat_catalog.yaml')
   for key, stats in content['categories'].items():
       ids = [s['id'] for s in stats]
       assert len(ids) == len(set(ids)), f'{key}に重複ID: {ids}'
       print(key, len(ids), '件、重複なし')
   "
   ```
5. キャッシュ・`--refresh`の動作確認
   - 再実行し、`既存のYAMLを使用します（再取得スキップ）`とログに出ること
   - `--refresh`を付けて再実行し、`fetched_at`が更新されること
6. `data_registry.json`に`pitcher`/`batter`それぞれの`stat_catalog.yaml`/`stat_catalog.meta.json`の`writers`/`readers`が記録されていることを確認（手動編集はしない）
