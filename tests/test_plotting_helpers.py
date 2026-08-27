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

from plot_helper import add_contours, make_2D_comparison2, make_corner_plot
from plot_helper import (
    APJ_DEFAULT_DPI,
    APJ_HALF_COLUMN_SQUARE_FIGSIZE,
    PLOT_DEFAULTS,
    TEX_POINTS_PER_INCH,
    configure_plot_defaults,
    figure_size_from_width,
    get_figure_preset,
    get_latex_font_size,
    tex_points_to_inches,
    use_apj_half_column_defaults,
)


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
        self.assertIsNone(fig.axes[0].get_legend())

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
    def test_make_2d_comparison2_defaults_multi_dataset_labels(self, _show):
        data1 = _bounded_data()
        data2 = _bounded_data()
        fig = make_2D_comparison2(
            [data1, data2],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
        )
        legend = fig.axes[0].get_legend()
        self.assertIsNotNone(legend)
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(labels, ["A", "B"])

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

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_respects_requested_axis_bounds(self, _show):
        rng = np.random.default_rng(42)
        data = {
            "x": rng.normal(0.0, 2.0, 500),
            "y": rng.normal(0.0, 2.0, 500),
        }
        fig = make_2D_comparison2(
            data,
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[-1, -2],
            b=[1, 2],
            scatter=False,
            boundary_method="reflection",
        )
        self.assertEqual(fig.axes[0].get_xlim(), (-1.0, 1.0))
        self.assertEqual(fig.axes[0].get_ylim(), (-2.0, 2.0))
        self.assertEqual(fig.axes[1].get_xlim(), (-1.0, 1.0))
        self.assertEqual(fig.axes[2].get_ylim(), (-2.0, 2.0))

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_filters_points_outside_requested_bounds(self, _show):
        data = {
            "x": np.array([-5.0, 0.0, 0.5, 10.0]),
            "y": np.array([0.0, 0.25, 0.75, 0.5]),
        }
        fig = make_2D_comparison2(
            data,
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0.0, 0.0],
            b=[1.0, 1.0],
            scatter=True,
            bins=2,
        )
        offsets = fig.axes[0].collections[0].get_offsets()
        self.assertEqual(len(offsets), 2)
        np.testing.assert_allclose(offsets[:, 0], [0.0, 0.5])
        np.testing.assert_allclose(offsets[:, 1], [0.25, 0.75])

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_scatter_mode_multiple_datasets(self, _show):
        data1 = _bounded_data()
        data2 = _bounded_data()
        fig = make_2D_comparison2(
            [data1, data2],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
            scatter=True,
            model_labels=["a", "b"],
        )
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].collections), 2)

    def test_add_contours_fill_alpha_tracks_quantiles(self):
        data = _bounded_data()
        fig, ax = plt.subplots()
        add_contours(
            ax,
            data["x"],
            data["y"],
            a=np.array([0.0, 0.0]),
            b=np.array([1.0, 1.0]),
            quantiles=[0.9, 0.5],
            boundary_method="reflection",
            fill=True,
        )
        alphas = sorted(
            {
                round(collection.get_alpha(), 3)
                for collection in ax.collections
                if collection.get_alpha() is not None and collection.get_alpha() > 0
            }
        )
        self.assertEqual(alphas, [0.1, 0.5])

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

    def test_make_corner_plot_accepts_hist_and_kde_linewidths(self):
        data = _bounded_data()
        fig, axes = make_corner_plot(
            [data],
            model_labels=[""],
            variables=["x", "y"],
            variable_labels=["x", "y"],
            kde=True,
            scatter=False,
            hist_linewidth=4,
            kde_linewidth=3,
            boundary_method="reflection",
            boundaries={"x": [0, 1], "y": [0, 1]},
            legend=False,
        )
        diag_ax = axes[0, 0]
        offdiag_ax = axes[1, 0]
        self.assertAlmostEqual(diag_ax.patches[0].get_linewidth(), 4.0)
        contour_linewidths = offdiag_ax.collections[0].get_linewidths()
        self.assertTrue(np.allclose(contour_linewidths, 3.0))
        self.assertIsNotNone(fig)

    def test_latex_font_size_helper(self):
        self.assertEqual(get_latex_font_size("scriptsize"), 7.0)
        self.assertEqual(get_latex_font_size("tiny"), 5.0)
        self.assertEqual(get_latex_font_size(9), 9.0)

    def test_apj_half_column_preset_values(self):
        preset = get_figure_preset("apj_half_column_square")
        self.assertEqual(preset["figsize"], APJ_HALF_COLUMN_SQUARE_FIGSIZE)
        self.assertEqual(preset["dpi"], APJ_DEFAULT_DPI)

    def test_tex_points_helpers(self):
        self.assertAlmostEqual(tex_points_to_inches(TEX_POINTS_PER_INCH), 1.0)
        self.assertEqual(figure_size_from_width(3.4, aspect_ratio=1.0), (3.4, 3.4))
        self.assertAlmostEqual(
            figure_size_from_width(246, aspect_ratio=0.5, units="pt")[0],
            246 / TEX_POINTS_PER_INCH,
        )

    def test_use_apj_half_column_defaults_updates_rcparams(self):
        with matplotlib.rc_context():
            use_apj_half_column_defaults(latex_font_size="tiny")
            self.assertEqual(tuple(matplotlib.rcParams["figure.figsize"]), APJ_HALF_COLUMN_SQUARE_FIGSIZE)
            self.assertEqual(matplotlib.rcParams["figure.dpi"], APJ_DEFAULT_DPI)
            self.assertEqual(matplotlib.rcParams["savefig.dpi"], APJ_DEFAULT_DPI)
            self.assertEqual(matplotlib.rcParams["font.size"], 5.0)
            self.assertEqual(matplotlib.rcParams["axes.labelsize"], 5.0)
            self.assertEqual(matplotlib.rcParams["legend.fontsize"], 5.0)

    def test_configure_plot_defaults_updates_global_instance(self):
        original = (
            PLOT_DEFAULTS.figsize,
            PLOT_DEFAULTS.dpi,
            PLOT_DEFAULTS.latex_font_size,
            PLOT_DEFAULTS.usetex,
        )
        try:
            with matplotlib.rc_context():
                defaults = configure_plot_defaults(figsize=(4.0, 2.0), dpi=123, latex_font_size="small")
                self.assertIs(defaults, PLOT_DEFAULTS)
                self.assertEqual(PLOT_DEFAULTS.figsize, (4.0, 2.0))
                self.assertEqual(PLOT_DEFAULTS.dpi, 123)
                self.assertEqual(PLOT_DEFAULTS.latex_font_size, "small")
                self.assertEqual(tuple(matplotlib.rcParams["figure.figsize"]), (4.0, 2.0))
                self.assertEqual(matplotlib.rcParams["figure.dpi"], 123)
        finally:
            PLOT_DEFAULTS.figsize, PLOT_DEFAULTS.dpi, PLOT_DEFAULTS.latex_font_size, PLOT_DEFAULTS.usetex = original
            PLOT_DEFAULTS.apply()

    def test_make_corner_plot_uses_global_default_figsize(self):
        original = (PLOT_DEFAULTS.figsize, PLOT_DEFAULTS.dpi)
        try:
            PLOT_DEFAULTS.figsize = APJ_HALF_COLUMN_SQUARE_FIGSIZE
            PLOT_DEFAULTS.dpi = APJ_DEFAULT_DPI
            fig, _ = make_corner_plot(
                [{"x": np.array([0.1, 0.2, 0.3]), "y": np.array([0.3, 0.4, 0.5])}],
                variables=["x", "y"],
                variable_labels=["x", "y"],
                kde=False,
                scatter=False,
                legend=False,
            )
            self.assertEqual(tuple(fig.get_size_inches()), APJ_HALF_COLUMN_SQUARE_FIGSIZE)
        finally:
            PLOT_DEFAULTS.figsize, PLOT_DEFAULTS.dpi = original
            PLOT_DEFAULTS.apply()

    @patch("matplotlib.pyplot.show")
    def test_make_2d_comparison2_uses_figure_preset(self, _show):
        fig = make_2D_comparison2(
            _bounded_data(),
            variables=["x", "y"],
            variable_labels=["x", "y"],
            a=[0, 0],
            b=[1, 1],
            figure_preset="apj_half_column_square",
        )
        self.assertEqual(tuple(fig.get_size_inches()), APJ_HALF_COLUMN_SQUARE_FIGSIZE)
        self.assertEqual(fig.dpi, APJ_DEFAULT_DPI)


if __name__ == "__main__":
    unittest.main()
