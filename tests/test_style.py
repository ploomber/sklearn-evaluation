import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn_evaluation.plot.style import apply_theme
from ploomber_core.exceptions import ValidationError
from matplotlib.testing.decorators import image_comparison


data = [[30, 25, 50, 20], [40, 23, 51, 17], [30, 25, 50, 20]]
X = np.arange(4)


def plot_bar():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    offset = 0
    for i in range(len(data)):
        ax.bar(X + offset, data[i], width=0.25)
        offset += 0.25

    return ax


def plot_bar_with_cmap(cmap=None):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    offset = 0
    for i in range(len(data)):
        color = mpl.colormaps.get_cmap(cmap)(float(i) / len(data))
        ax.bar(X + offset, data[i], width=0.25, color=color)
        offset += 0.25

    return ax


@pytest.mark.parametrize(
    "ax_style, cmap_style, expected_errors",
    [
        [
            "frame1",
            "gradient",
            [
                "Valid values are: 'frame', and 'no_frame'",
                "Did you mean 'frame', or 'no_frame'",
            ],
        ],
        ["", "gradient", ["Valid values are: 'frame', and 'no_frame'"]],
        [
            "no_frme",
            "monochromatic",
            [
                "Valid values are: 'frame', and 'no_frame'",
                "Did you mean 'frame', or 'no_frame'",
            ],
        ],
        [
            "frame",
            "gradient1",
            [
                "Valid values are: 'gradient', and 'monochromatic'",
                "Did you mean 'gradient'",
            ],
        ],
        ["frame", "", ["Valid values are: 'gradient', and 'monochromatic'"]],
        [
            "no_frame",
            "monochromaticc",
            [
                "Valid values are: 'gradient', and 'monochromatic'",
                "Did you mean 'monochromatic'",
            ],
        ],
        [
            "frame1",
            "gradient1",
            [
                "Valid values are: 'frame', and 'no_frame'",
                "Did you mean 'frame', or 'no_frame'",
            ],
        ],
        [None, "gradient1", ["Valid values are: 'frame', and 'no_frame'"]],
    ],
)
def test_apply_theme_bad_input(ax_style, cmap_style, expected_errors):
    @apply_theme(ax_style=ax_style, cmap_style=cmap_style)
    def with_style():
        plot_bar()

    with pytest.raises(ValidationError) as err:
        with_style()

    _err_message = str(err.value)

    for expected in expected_errors:
        assert expected in _err_message


@image_comparison(baseline_images=["no_style"], extensions=["png"], remove_text=False)
def test_no_style():
    plot_bar()


@image_comparison(
    baseline_images=["default_style"], extensions=["png"], remove_text=False
)
def test_default_style():
    @apply_theme()
    def with_style():
        plot_bar()

    with_style()


@image_comparison(
    baseline_images=["custom_cmap_style"], extensions=["png"], remove_text=False
)
def test_custom_cmap():
    plot_bar_with_cmap(cmap="plasma")


@image_comparison(
    baseline_images=["monochromatic_cmap_style"], extensions=["png"], remove_text=False
)
def test_default_style_cmap():
    @apply_theme()
    def with_style():
        plot_bar_with_cmap()

    with_style()


@image_comparison(
    baseline_images=["gradient_cmap_style"], extensions=["png"], remove_text=False
)
def test_gradient_style_cmap():
    @apply_theme(cmap_style="gradient")
    def with_style():
        plot_bar_with_cmap()

    with_style()


@image_comparison(
    baseline_images=["monochromatic_cmap_style"], extensions=["png"], remove_text=False
)
def test_monochromatic_style_cmap():
    @apply_theme(cmap_style="monochromatic")
    def with_style():
        plot_bar_with_cmap()

    with_style()


@image_comparison(
    baseline_images=["frame_monochromatic_cmap_style"],
    extensions=["png"],
    remove_text=False,
)
def test_frame_monochromatic_style_cmap():
    @apply_theme(ax_style="frame")
    def with_style():
        plot_bar_with_cmap()

    with_style()


@image_comparison(
    baseline_images=["frame_gradient_cmap_style"], extensions=["png"], remove_text=False
)
def test_frame_gradient_style_cmap():
    @apply_theme(ax_style="frame", cmap_style="gradient")
    def with_style():
        plot_bar_with_cmap()

    with_style()


@image_comparison(
    baseline_images=["custom_style"], extensions=["png"], remove_text=False
)
def test_custom_style():
    @apply_theme()
    def with_style(cmap):
        ax = plot_bar_with_cmap(cmap)

        ax.grid(color="#c6c6c6", linestyle="--", linewidth=0.5)

    with_style(cmap="viridis")
