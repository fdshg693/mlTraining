# 線形解析の再実装を試みる
# シンプルな2次元データのみを対象にする
# (x1,y1), (x2,y2), ... のようなデータを想定する
# 入力として X = [[x1], [x2], ...]、出力として y = [y1, y2, ...] のような形を想定する

# 最小二乗法を使った線形回帰の実装を試みる
# 傾きは (x-x_mean) と (y-y_mean) の積の平均を (x-x_mean) の二乗の平均で割ることで求められる
# 切片は y_mean - slope * x_mean で求められる

# 実装アルゴリズム
# 1. X と y の平均を計算する
# 2. 傾きを計算する
# 3. 切片を計算する
# 4. 予測関数を定義する

# 外部ライブラリには依存しない（スクラッチ実装のため純 Python のみ）


def _flatten_x(X):
    """X = [[x1], [x2], ...] を [x1, x2, ...] に変換する。

    すでに [x1, x2, ...] の形（スカラーのリスト）ならそのまま使う。
    """
    flat = []
    for row in X:
        if isinstance(row, (list, tuple)):
            if len(row) != 1:
                raise ValueError("各サンプルの特徴量は 1 個（2 次元データ）のみ対応です")
            flat.append(float(row[0]))
        else:
            flat.append(float(row))
    return flat


def _weighted_mean(values, weights):
    return sum(w * v for v, w in zip(values, weights)) / sum(weights)


class SimpleLinearRegression:
    """最小二乗法による単回帰（y = slope * x + intercept）。

    sklearn の LinearRegression に倣って fit / predict を持つ。
    """

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y, sample_weight=None):
        xs = _flatten_x(X)
        ys = [float(v) for v in y]

        if len(xs) != len(ys):
            raise ValueError("X と y の長さが一致しません")
        if len(xs) < 2:
            raise ValueError("回帰には 2 点以上のデータが必要です")

        # 重み（sample_weight）の準備
        #   None なら全サンプル重み 1 ＝ 通常の最小二乗法と同じ挙動になる
        if sample_weight is None:
            ws = [1.0] * len(xs)
        else:
            ws = [float(w) for w in sample_weight]
            if len(ws) != len(xs):
                raise ValueError("sample_weight の長さが X と一致しません")
            if any(w < 0 for w in ws):
                raise ValueError("sample_weight に負の値は指定できません")
            if sum(ws) == 0:
                raise ValueError("sample_weight の合計が 0 です")

        # 1. 重み付き平均を計算する
        x_mean = _weighted_mean(xs, ws)
        y_mean = _weighted_mean(ys, ws)

        # 2. 傾きを計算する
        #    slope = Σ w(x-x_mean)(y-y_mean) / Σ w(x-x_mean)^2
        numerator = sum(w * (x - x_mean) * (yi - y_mean) for x, yi, w in zip(xs, ys, ws))
        denominator = sum(w * (x - x_mean) ** 2 for x, w in zip(xs, ws))
        if denominator == 0:
            raise ValueError("X の値がすべて同じため傾きを決定できません")
        self.slope = numerator / denominator

        # 3. 切片を計算する
        self.intercept = y_mean - self.slope * x_mean
        return self

    def predict(self, X):
        if self.slope is None or self.intercept is None:
            raise ValueError("predict の前に fit を呼んでください")
        xs = _flatten_x(X)
        return [self.slope * x + self.intercept for x in xs]


if __name__ == "__main__":
    # y = 2x + 1 の関係を持つデータで動作確認する
    X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]

    model = SimpleLinearRegression().fit(X, y)
    print(f"slope={model.slope:.4f}, intercept={model.intercept:.4f}")
    print("predict([[5]]) =", model.predict([[5.0]]))

    # 重みづけの動作確認
    # 最後の点 (4, 9) を外れ値 (4, 100) に差し替える
    Xw = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    yw = [1.0, 3.0, 5.0, 7.0, 100.0]

    # 重みなし: 外れ値に引っ張られて傾き・切片がずれる
    m_plain = SimpleLinearRegression().fit(Xw, yw)
    print(f"\n[外れ値あり・重みなし] slope={m_plain.slope:.4f}, intercept={m_plain.intercept:.4f}")

    # 外れ値の重みを小さくすると、元の y=2x+1 にほぼ戻る
    weights = [1.0, 1.0, 1.0, 1.0, 0.001]
    m_weighted = SimpleLinearRegression().fit(Xw, yw, sample_weight=weights)
    print(f"[外れ値あり・重みづけ] slope={m_weighted.slope:.4f}, intercept={m_weighted.intercept:.4f}")