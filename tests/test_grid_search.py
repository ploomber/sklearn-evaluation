import sys
from functools import partial

import pytest
from matplotlib.testing.decorators import image_comparison as _image_comparison, cleanup
from sklearn_evaluation import plot
from ploomber_core.exceptions import ValidationError


image_comparison = partial(
    _image_comparison,
    tol=0.6 if sys.version_info.minor == 6 else 0,
    remove_text=True,
    extensions=["png"],
)


@image_comparison(baseline_images=["single_numeric_line"])
def test_single_numeric_line(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="line")


@image_comparison(baseline_images=["single_numeric_line_unsorted"])
def test_single_numeric_line_no_sort(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="line", sort=False)


@image_comparison(baseline_images=["single_numeric_line"])
def test_single_numeric_line_with_tuple(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="line")


@image_comparison(baseline_images=["single_numeric_bar"])
def test_single_numeric_bar(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar")


@image_comparison(baseline_images=["single_numeric_bar_unsorted"])
def test_single_numeric_bar_unsorted(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar", sort=False)


@image_comparison(baseline_images=["single_categorical_line"])
def test_single_categorial_line(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="line")


@image_comparison(baseline_images=["single_categorical_line_unsorted"])
def test_single_categorial_line_unsorted(grid_search_3_params):
    change = "criterion"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="line", sort=False)


@image_comparison(baseline_images=["single_categorical_bar"])
def test_single_categorial_bar(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar")


@image_comparison(baseline_images=["single_categorical_bar_unsorted"])
def test_single_categorial_bar_unsorted(grid_search_3_params):
    change = "criterion"
    plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar", sort=False)


@image_comparison(baseline_images=["single_numeric_partially_restricted"])
def test_single_numeric_partially_restricted(grid_search_3_params):
    change = "n_estimators"
    subset = {"max_features": ["sqrt"]}
    plot.grid_search(grid_search_3_params.cv_results_, change, subset, kind="bar")


@image_comparison(
    baseline_images=["single_numeric_partially_restricted_unsorted"],
    extensions=["png"],
    remove_text=True,
)
def test_single_numeric_partially_restricted_unsorted(grid_search_3_params):
    change = "n_estimators"
    subset = {"max_features": "sqrt"}
    plot.grid_search(
        grid_search_3_params.cv_results_, change, subset, kind="bar", sort=False
    )


@image_comparison(baseline_images=["single_numeric_restricted_single"])
def test_single_numeric_restricted_single(grid_search_3_params):
    change = "n_estimators"
    subset = {"max_features": "sqrt", "criterion": "gini"}
    plot.grid_search(grid_search_3_params.cv_results_, change, subset, kind="bar")


@image_comparison(baseline_images=["single_numeric_restricted_multi"])
def test_single_numeric_restricted_multi(grid_search_3_params):
    change = "n_estimators"
    subset = {"max_features": ["sqrt", "log2"], "criterion": "gini"}
    plot.grid_search(grid_search_3_params.cv_results_, change, subset, kind="bar")


@image_comparison(baseline_images=["single_numeric_restricted_multi_unsorted"])
def test_single_numeric_restricted_multi_unsorted(grid_search_3_params):
    change = "n_estimators"
    subset = {"max_features": ["sqrt", "log2"], "criterion": "gini"}
    plot.grid_search(
        grid_search_3_params.cv_results_, change, subset, kind="bar", sort=False
    )


@image_comparison(baseline_images=["double"])
def test_double_ignores_kind_line(grid_search_3_params):
    change = ("n_estimators", "criterion")
    subset = {"max_features": "sqrt"}
    plot.grid_search(grid_search_3_params.cv_results_, change, subset, kind="line")


@image_comparison(baseline_images=["double"])
def test_double_ignores_kind_bar(grid_search_3_params):
    change = ("n_estimators", "criterion")
    subset = {"max_features": "sqrt"}
    plot.grid_search(grid_search_3_params.cv_results_, change, subset, kind="bar")


@pytest.mark.parametrize(
    "change, expected_suggestions",
    [
        ["some_n_estimators", ["n_estimators"]],
        ["erion", ["criterion"]],
        ["max_feas", ["max_features"]],
        [("some_n_estimators", "n_estimators", "criterion"), ["n_estimators"]],
        [("n_estimators1", "criterion1"), ["n_estimators", "criterion"]],
        [("n_estimators1", ""), ["n_estimators"]],
    ],
)
def test_invalid_change_param_with_suggestions(
    grid_search_3_params, change, expected_suggestions
):
    with pytest.raises(ValidationError) as e:
        plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar")

    assert "Did you mean" in str(e.value)

    for suggestion in expected_suggestions:
        assert suggestion in str(e.value)


@pytest.mark.parametrize(
    "change", ["123", "a", "", ("", "a"), ("some-text", ""), ("some-text", "more")]
)
def test_invalid_change_param_without_suggestions(grid_search_3_params, change):
    with pytest.raises(ValidationError) as e:
        plot.grid_search(grid_search_3_params.cv_results_, change, kind="bar")

    assert "Did you mean" not in str(e.value)


@pytest.mark.parametrize(
    "kind, expected_suggestions",
    [["bar1", ["bar"]], ["liner", ["line"]], ["lien", ["line"]]],
)
def test_invalid_kind_param_with_suggestions(
    grid_search_3_params, kind, expected_suggestions
):
    with pytest.raises(ValidationError) as e:
        plot.grid_search(
            grid_search_3_params.cv_results_, change="n_estimators", kind=kind
        )

    assert "Did you mean" in str(e.value)

    for suggestion in expected_suggestions:
        assert suggestion in str(e.value)


@pytest.mark.parametrize("kind", ["donut", "pie"])
def test_invalid_kind_param_without_suggestions(grid_search_3_params, kind):
    with pytest.raises(ValidationError) as e:
        plot.grid_search(
            grid_search_3_params.cv_results_, change="n_estimators", kind=kind
        )

    assert "Did you mean" not in str(e.value)


# API tests


@cleanup
def test_list_with_len_three_raises_exception(grid_search_3_params):
    axis = ["a", "b", "c"]
    with pytest.raises(ValidationError):
        plot.grid_search(grid_search_3_params.cv_results_, axis)


@cleanup
def test_none_change_raises_exception(
    grid_search_3_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.grid_search(grid_search_3_params.cv_results_, None)


@cleanup
def test_can_send_tuple_len_one(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change)


@cleanup
def test_can_send_string(grid_search_3_params):
    change = "n_estimators"
    plot.grid_search(grid_search_3_params.cv_results_, change)


@cleanup
def test_raise_exception_when_parameter_set_is_not_fully_specified(
    grid_search_3_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = ("n_estimators", "criterion")
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=None)


@cleanup
def test_subset_can_be_none_when_parameter_set_is_fully_specified(grid_search_2_params):
    change = ("n_estimators", "criterion")
    plot.grid_search(grid_search_2_params.cv_results_, change=change, subset=None)


@cleanup
def test_parameter_has_none_value_double(grid_search_param_with_none):
    plot.grid_search(
        grid_search_param_with_none.cv_results_,
        change=("max_depth", "criterion"),
        subset=None,
    )


@cleanup
def test_raise_exception_when_parameter_has_none_value_single(
    grid_search_param_with_none,
):
    with pytest.raises(TypeError):
        plot.grid_search(
            grid_search_param_with_none.cv_results_, change="criterion", subset=None
        )


@cleanup
def test_parameter_has_none_value_single(grid_search_param_with_none):
    plot.grid_search(
        grid_search_param_with_none.cv_results_,
        change="criterion",
        subset=None,
        sort=False,
    )


@cleanup
def test_raise_exception_when_parameter_subset_matches_more_than_one_group(
    grid_search_4_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = ("n_estimators", "criterion")
        subset = {"min_samples_split": 2}
        plot.grid_search(grid_search_4_params.cv_results_, change=change, subset=subset)


@cleanup
def test_raise_exception_when_parameter_does_not_exist(grid_search_3_params):
    with pytest.raises(ValidationError):
        change = "this_is_not_a_parameter"
        subset = {"criterion": "gini", "max_features": "sqrt"}
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=subset)


@cleanup
def test_raise_exception_when_parameter_does_not_exist_double(grid_search_3_params):
    with pytest.raises(ValidationError):
        change = ("n_estimators", "this_is_not_a_parameter")
        subset = {"criterion": "gini", "max_features": "sqrt"}
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=subset)


@cleanup
def test_raise_exception_when_invalid_value_in_subset_double(
    grid_search_3_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = ("n_estimators", "max_features")
        subset = {"criterion": "not_a_value"}
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=subset)


@cleanup
def test_raise_exception_when_invalid_value_in_subset(
    grid_search_3_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = "n_estimators"
        subset = {"criterion": "not_a_value"}
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=subset)


@cleanup
def test_raise_exception_when_passing_repeated_parameters(
    grid_search_3_params, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = ["n_estimators", "n_estimators"]
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=None)


@cleanup
def test_none_parameter_wit(grid_search_3_params, ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        change = ["n_estimators", "n_estimators"]
        plot.grid_search(grid_search_3_params.cv_results_, change=change, subset=None)
