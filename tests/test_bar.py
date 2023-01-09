from functools import partial
from matplotlib.testing.decorators import image_comparison as _image_comparison

from sklearn_evaluation.plot import _example


image_comparison = partial(
    _image_comparison,
    remove_text=True,
    extensions=["png"],
)


@image_comparison(baseline_images=["mybar-raw-data"])
def test_mybar_from_raw_data():
    _example.MyBar.from_raw_data(
        ["banana", "banana", "apple", "pineapple", "apple"], color="lightblue"
    )


@image_comparison(baseline_images=["mybar"])
def test_mybar():
    _example.MyBar.from_raw_data({"banana": 10, "apple": 5}, color="lightgreen")


@image_comparison(baseline_images=["mybar-add"])
def test_mybar_add():
    # NOTE: we use the constructor because from_raw_data calls
    # .plot() internally, causing to display the individual plots
    # whihc we do not want to test
    one = _example.MyBar({"apple": 2, "banana": 3})
    another = _example.MyBar({"apple": 2, "banana": 5})
    one + another


@image_comparison(baseline_images=["mybar-sub"])
def test_mybar_sub():
    # NOTE: we use the constructor because from_raw_data calls
    # .plot() internally, causing to display the individual plots
    # whihc we do not want to test
    one = _example.MyBar({"apple": 2, "banana": 3})
    another = _example.MyBar({"apple": 1, "banana": 5})
    one - another


@image_comparison(baseline_images=["my-bar-functional"])
def test_my_bar_functional():
    _example.my_bar(
        ["banana", "banana", "apple", "pineapple", "apple"], color="lightblue"
    )
