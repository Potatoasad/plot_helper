import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_helper import make_2D_comparison2, make_corner_plot


OUTPUT_DIR = "/Users/ahussain/Documents/GitHub/plot_helper/examples"


def make_dict(x):
    return {k: x[k].values for k in x.columns}


def main():
    rng = np.random.default_rng(12345)
    df1 = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["x1", "x2", "x3"])
    df2 = pd.DataFrame(rng.normal(size=(1000, 3)) + 1, columns=["x1", "x2", "x3"])

    fig, axes = make_corner_plot(
        all_data=[make_dict(df1), make_dict(df2)],
        kde=True,
        scatter=False,
    )
    fig.savefig(f"{OUTPUT_DIR}/example_inferred_variables.png", bbox_inches="tight")
    plt.close(fig)

    fig = make_2D_comparison2(
        df1,
        variables=["x1", "x2"],
        variable_labels=[r"$x_1$", r"$x_2$"],
        a=[-3, -3],
        b=[3, 3],
    )
    fig.savefig(f"{OUTPUT_DIR}/example_2D_comparison_single.png", bbox_inches="tight")
    plt.close(fig)

    fig = make_2D_comparison2(
        [df1, df2],
        variables=["x1", "x2"],
        variable_labels=[r"$x_1$", r"$x_2$"],
        a=[-3, -3],
        b=[3, 3],
        model_labels=["first", "second"],
        scatter=False,
        bins=[20, 20],
        boundary_method="reflection",
    )
    fig.savefig(f"{OUTPUT_DIR}/example_2D_comparison_multi.png", bbox_inches="tight")
    plt.close(fig)

    fig = make_2D_comparison2(
        [df1, df2],
        variables=["x1", "x2"],
        variable_labels=[r"$x_1$", r"$x_2$"],
        a=[-3, -3],
        b=[3, 3],
        scatter=True,
        model_labels=["first", "second"],
    )
    fig.savefig(f"{OUTPUT_DIR}/example_2D_comparison_scatter.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
