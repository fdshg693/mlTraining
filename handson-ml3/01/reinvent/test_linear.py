# reinvent/linear.py の SimpleLinearRegression のテスト
#
# 実行方法（リポジトリのどこからでも可）:
#   python -m unittest 01_the_machine_learning_landscape/reinvent/test_linear.py
# または reinvent ディレクトリ内で:
#   python -m unittest test_linear

import sys
import unittest
from pathlib import Path

# このファイルと同じディレクトリの linear.py を import できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear import SimpleLinearRegression, _flatten_x


class TestSimpleLinearRegression(unittest.TestCase):
    def test_perfect_line(self):
        # y = 2x + 1 を完全に復元できるか
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        model = SimpleLinearRegression().fit(X, y)
        self.assertAlmostEqual(model.slope, 2.0, places=9)
        self.assertAlmostEqual(model.intercept, 1.0, places=9)

    def test_predict(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        model = SimpleLinearRegression().fit(X, y)
        preds = model.predict([[5.0], [10.0]])
        self.assertAlmostEqual(preds[0], 11.0, places=9)
        self.assertAlmostEqual(preds[1], 21.0, places=9)

    def test_negative_slope(self):
        # y = -3x + 5
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, -1.0, -4.0]
        model = SimpleLinearRegression().fit(X, y)
        self.assertAlmostEqual(model.slope, -3.0, places=9)
        self.assertAlmostEqual(model.intercept, 5.0, places=9)

    def test_noisy_data_least_squares(self):
        # ノイズありデータ。手計算の最小二乗解と一致するか確認する。
        # データ: (1,1), (2,2), (3,2)
        #   x_mean=2, y_mean=5/3
        #   slope = Σ(x-2)(y-y_mean) / Σ(x-2)^2 = 1/2
        #   intercept = 5/3 - (1/2)*2 = 2/3
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 2.0]
        model = SimpleLinearRegression().fit(X, y)
        self.assertAlmostEqual(model.slope, 0.5, places=9)
        self.assertAlmostEqual(model.intercept, 2.0 / 3.0, places=9)

    def test_accepts_flat_x(self):
        # X が [x1, x2, ...] のスカラー列でも動くこと
        model = SimpleLinearRegression().fit([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(model.slope, 2.0, places=9)
        self.assertAlmostEqual(model.intercept, 1.0, places=9)

    def test_fit_returns_self(self):
        model = SimpleLinearRegression()
        self.assertIs(model.fit([[0], [1]], [0, 1]), model)

    def test_predict_before_fit_raises(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.predict([[1.0]])

    def test_mismatched_lengths_raise(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[0], [1], [2]], [0, 1])

    def test_too_few_points_raise(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[0]], [0])

    def test_constant_x_raises(self):
        # X がすべて同じ値だと傾きが決まらない
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[2.0], [2.0], [2.0]], [1.0, 2.0, 3.0])

    def test_multifeature_x_raises(self):
        # 2 次元データ専用なので特徴量が複数あるとエラー
        with self.assertRaises(ValueError):
            _flatten_x([[1.0, 2.0], [3.0, 4.0]])


class TestWeightedRegression(unittest.TestCase):
    def test_none_weight_matches_unweighted(self):
        # sample_weight=None は明示しない場合と完全に同じ結果になる
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 2.0]
        m1 = SimpleLinearRegression().fit(X, y)
        m2 = SimpleLinearRegression().fit(X, y, sample_weight=None)
        self.assertAlmostEqual(m1.slope, m2.slope, places=9)
        self.assertAlmostEqual(m1.intercept, m2.intercept, places=9)

    def test_equal_weights_match_unweighted(self):
        # 全サンプル同じ重みなら通常の最小二乗法と一致する（定数倍でも不変）
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 2.0]
        plain = SimpleLinearRegression().fit(X, y)
        weighted = SimpleLinearRegression().fit(X, y, sample_weight=[5.0, 5.0, 5.0])
        self.assertAlmostEqual(plain.slope, weighted.slope, places=9)
        self.assertAlmostEqual(plain.intercept, weighted.intercept, places=9)

    def test_integer_weight_equals_duplication(self):
        # 整数の重み w は、その点を w 回複製したデータと等価になるはず
        Xw = [[0.0], [1.0], [2.0]]
        yw = [0.0, 1.0, 10.0]
        weighted = SimpleLinearRegression().fit(Xw, yw, sample_weight=[1.0, 1.0, 2.0])

        Xdup = [[0.0], [1.0], [2.0], [2.0]]
        ydup = [0.0, 1.0, 10.0, 10.0]
        duplicated = SimpleLinearRegression().fit(Xdup, ydup)

        self.assertAlmostEqual(weighted.slope, duplicated.slope, places=9)
        self.assertAlmostEqual(weighted.intercept, duplicated.intercept, places=9)

    def test_downweighting_outlier_recovers_line(self):
        # 外れ値の重みを小さくすると元の y=2x+1 にほぼ戻る
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 3.0, 5.0, 7.0, 100.0]  # 最後が外れ値
        model = SimpleLinearRegression().fit(
            X, y, sample_weight=[1.0, 1.0, 1.0, 1.0, 1e-9]
        )
        self.assertAlmostEqual(model.slope, 2.0, places=6)
        self.assertAlmostEqual(model.intercept, 1.0, places=6)

    def test_mismatched_weight_length_raises(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[0], [1], [2]], [0, 1, 2], sample_weight=[1.0, 1.0])

    def test_negative_weight_raises(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[0], [1], [2]], [0, 1, 2], sample_weight=[1.0, -1.0, 1.0])

    def test_zero_total_weight_raises(self):
        model = SimpleLinearRegression()
        with self.assertRaises(ValueError):
            model.fit([[0], [1], [2]], [0, 1, 2], sample_weight=[0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
