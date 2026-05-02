import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from plot_helper import make_2D_comparison2, make_corner_plot


def _bounded_data(n=200):
    rng = np.random.default_rng(1234)
    x = np.clip(rng.normal(0.4, 0.12, n), 0.0, 1.0)
    y = np.clip(rng.normal(0.6, 0.15, n), 0.0, 1.0)
    z = np.clip(rng.normal(0.5, 0.10, n), 0.0, 1.0)
    return {"x": x, "y": y, "z": z}


def _assert_contours_within_bounds(test_case, ax, x_bounds, y_bounds):
    has_vertices = False
    for collection in ax.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            if len(vertices) == 0:
                continue
            has_vertices = True
            test_case.assertGreaterEqual(vertices[:, 0].min(), x_bounds[0] - 1e-8)
            test_case.assertLessEqual(vertices[:, 0].max(), x_bounds[1] + 1e-8)
            test_case.assertGreaterEqual(vertices[:, 1].min(), y_bounds[0] - 1e-8)
            test_case.assertLessEqual(vertices[:, 1].max(), y_bounds[1] + 1e-8)
    test_case.assertTrue(has_vertices)


class PlottingHelpersSmokeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_accepts_single_dataframe(self, _show):
        if pd is None:
            self.skipTest("pandas is not installed")

        data = _bounded_data()
        frame = pd.DataFrame(data)
        fig = make_2D_comparison2(
            frame,
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
        )
        self.assertEqual(len(fig.axes), 3)

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_accepts_list_and_custom_bins(self, _show):
        data1 = _bounded_data()
        data2 = _bounded_data()
        fig = make_2D_comparison2(
            [data1, data2],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
            bins=[12, 8],
            model_labels=["a", "b"],
        )
        self.assertEqual(len(fig.axes), 3)

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_scatter_mode(self, _show):
        fig = make_2D_comparison2(
            _bounded_data(),
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
            scatter=True,
            alpha=0.2,
        )
        self.assertEqual(len(fig.axes[0].collections), 1)

    def test_make_corner_plot_defaults_model_labels_and_variables(self):
        data1 = {"x": np.array([0.1, 0.2, 0.3]), "y": np.array([0.3, 0.4, 0.5])}
        data2 = {"y": np.array([0.2, 0.3, 0.4]), "z": np.array([0.6, 0.7, 0.8])}
        fig, axes = make_corner_plot(
            [data1, data2],
            kde=False,
            scatter=False,
            legend=True,
        )
        self.assertEqual(axes.shape, (3, 3))
        self.assertEqual(axes[-1, 0].get_xlabel(), "x")
        self.assertEqual(axes[-1, 1].get_xlabel(), "y")
        self.assertEqual(axes[-1, 2].get_xlabel(), "z")
        self.assertIsNotNone(fig)

    def test_make_corner_plot_boundary_bias_true_uses_bounded_path(self):
        data = _bounded_data()
        fig, axes = make_corner_plot(
            [data],
            model_labels=[""],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            kde=True,
            scatter=False,
            boundary_bias=True,
            boundaries={"x": [0, 1], "y": [0, 1]},
            legend=False,
        )
        ax = axes[1, 0]
        self.assertGreater(len(ax.collections), 0)
        _assert_contours_within_bounds(self, ax, (0.0, 1.0), (0.0, 1.0))
        self.assertIsNotNone(fig)

    def test_make_corner_plot_reflection_boundary_method(self):
        data = _bounded_data()
        fig, axes = make_corner_plot(
            [data],
            model_labels=[""],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            kde=True,
            scatter=False,
            boundary_method="reflection",
            boundaries={"x": [0, 1], "y": [0, 1]},
            legend=False,
        )
        ax = axes[1, 0]
        self.assertGreater(len(ax.collections), 0)
        _assert_contours_within_bounds(self, ax, (0.0, 1.0), (0.0, 1.0))
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
