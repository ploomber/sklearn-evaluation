"""
Plotting functions for classifier models
"""

import json
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from collections import defaultdict

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn_evaluation import __version__
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.plot.plot import AbstractPlot
from ploomber_core.dependencies import requires
from ploomber_core.exceptions import modify_exceptions


def _confusion_matrix_validate_predictions(y_true, y_pred, target_names):
    """Validate input prediction data for a confusion matrix"""
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion " "matrix")

    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)

    if target_names and (expected_len != len(target_names)):
        raise ValueError(
            (
                "Data contains {} different values, but target"
                " names contains {} values.".format(expected_len, len(target_names))
            )
        )

    if not target_names:
        values = list(values)
        values.sort()
        target_names = ["Class {}".format(v) for v in values]

    return target_names


def _validate_test_dataset(X_test, feature_names):
    ncols = 0
    if isinstance(X_test, np.ndarray):
        ncols = X_test.shape[1]
    elif isinstance(X_test, list):
        ncols = len(X_test[0])
    if feature_names and len(feature_names) != ncols:
        raise ValueError(
            (
                "Data contains {} different columns, but feature"
                " names contains {} values.".format(ncols, len(feature_names))
            )
        )
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(ncols)]
    return pd.DataFrame(X_test, columns=feature_names)


def _confusion_matrix(y_true, y_pred, normalize):
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm


def _quadrant_wise_indices(y_true, y_pred):
    unique_targets = set(y_true).union(set(y_pred))
    combinations = list(product(unique_targets, repeat=2))
    indices = {}
    for c in combinations:
        indices[c] = []
    for ind, (x, y) in enumerate(zip(y_true, y_pred)):
        indices[(x, y)].append(ind)
    return indices


def _is_long_number(num):
    if "." in str(num):
        split_by_decimal = str(num).split(".")
        if len(split_by_decimal[0]) > 10 or len(split_by_decimal[1]) > 10:
            return True
    return False


def _convert_to_scientific(data):
    scientific_notation_data = {}
    for key, value in data.items():
        if "sampled" in key or "metric" in key:
            new_value = []
            for quadrant in value:
                new_quadrant_value = []
                for v in quadrant:
                    if (
                        isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and _is_long_number(v)
                    ):
                        new_quadrant_value.append(
                            np.format_float_scientific(v, exp_digits=2, precision=3)
                        )
                    else:
                        new_quadrant_value.append(v)
                new_value.append(new_quadrant_value)
            scientific_notation_data[key] = new_value
        else:
            scientific_notation_data[key] = value
    return scientific_notation_data


def _quadrant_interactive_data(X_test, quadrants, nsample):
    if X_test is not None:
        columns = X_test.columns.tolist()

        sampled_rows = defaultdict(list)

        metric_rows = defaultdict(list)

        for (_, _), quadrant_indices in sorted(quadrants.items()):
            select_indices = random.sample(
                quadrant_indices, min(len(quadrant_indices), nsample)
            )
            column_data, describe = _data_by_column(
                X_test.iloc[select_indices], columns
            )
            for key, value in column_data.items():
                sampled_rows[f"{key} sampled"].append(value)
            column_data, describe = _data_by_column(
                X_test.iloc[quadrant_indices], columns
            )
            for key, value in column_data.items():
                describe_data = describe[key].tolist()

                describe_data = [
                    int(x) if isinstance(x, np.int64) else x for x in describe_data
                ]
                metric_rows[f"{key} metric"].append(describe_data)
            metric_rows["quadrant statistics name"].append(describe.index.tolist())
        quadrant_data = _convert_to_scientific({**sampled_rows, **metric_rows})
        return dict(quadrant_data)
    return None


def _data_by_column(data, columns):
    describe = data.describe(include="all").fillna("NA")
    return {col: data[col].tolist() for col in columns}, describe


def _cm_plot_data(cm, targets):
    cm = [y for i in cm for y in i]
    roll = list(product(targets, repeat=2))
    actual_data = []
    predicted_data = []
    cm_data = []

    for i in range(len(roll)):
        actual_data.append(roll[i][0])
        predicted_data.append(roll[i][1])
        cm_data.append(cm[i])

    return {
        "actual": actual_data,
        "predicted": predicted_data,
        "confusion_matrix": cm_data,
    }


def _plot_cm_chart(df, selection, alt):
    if selection is not None:
        color = alt.condition(
            selection,
            "confusion_matrix:N",
            alt.value("lightgray"),
            scale=alt.Scale(scheme="oranges"),
        )
    else:
        color = alt.Color(scale=alt.Scale(scheme="oranges"))

    heatmap = (
        alt.Chart(df, title="Confusion Matrix")
        .mark_rect()
        .encode(
            alt.X("predicted:N"),
            alt.Y("actual:N"),
            color=color,
        )
    )

    text = (
        alt.Chart(df, title="Confusion Matrix")
        .mark_text(baseline="middle", fontSize=25, fontWeight="bold")
        .encode(
            x="predicted:N",
            y="actual:N",
            text="confusion_matrix:N",
            color=alt.condition(
                alt.datum.confusion_matrix > 0,
                alt.value("black"),
                alt.value("black"),
            ),
        )
    )
    cm_chart = (heatmap + text).properties(width=600, height=480)
    return cm_chart


def _plot_sample_data_chart(df, selection, columns, alt):
    # Base chart for sampled data tables
    table_base = (
        alt.Chart(df)
        .mark_text(align="center", baseline="top")
        .add_params(selection)
        .transform_filter(selection)
        .properties(width=120, height=150)
    )

    table_charts = [
        table_base.encode(text=f"{col}:N").properties(title=col.replace(" sampled", ""))
        for col in columns
    ]
    return alt.hconcat(*table_charts, spacing=0).properties(title="Sample Observations")


def _plot_data_statistics_chart(df, selection, columns, alt):
    metric_table_base = (
        alt.Chart(df)
        .mark_text(align="center", baseline="line-top", fontSize=14, fill="#09156A")
        .add_params(selection)
        .transform_filter(selection)
        .properties(width=120, height=290)
    )

    name_column = metric_table_base.encode(text="quadrant statistics name:N")

    metric_table_charts = [
        metric_table_base.encode(text=f"{col}:N").properties(
            title=col.replace(" metric", "")
        )
        for col in columns
    ]

    metric_text = alt.hconcat(name_column, *metric_table_charts, spacing=0).properties(
        title="Quadrant Statistics"
    )

    metric_text.configure_title(fontSize=20)
    return metric_text


class InteractiveConfusionMatrix(AbstractPlot):
    """
    Plot interactive confusion matrix.

    Notes
    -----
    .. versionadded:: 0.11.3
    """

    @SKLearnEvaluationLogger.log(
        feature="plot", action="interactive-confusion-matrix-init"
    )
    @modify_exceptions
    def __init__(self, cm, *, target_names=None, interactive_data=None):
        self.cm = cm
        self.target_names = target_names
        self.interactive_data = interactive_data

    @requires(["altair"])
    def plot(self):
        import altair as alt

        cm_data = _cm_plot_data(self.cm, self.target_names)

        if self.interactive_data is not None:
            sampled_columns = [
                col for col in list(self.interactive_data.keys()) if "sampled" in col
            ]

            metric_columns = [
                col for col in list(self.interactive_data.keys()) if "metric" in col
            ]

            cm_data = {**cm_data, **self.interactive_data}

        df = pd.DataFrame(data=cm_data)

        if self.interactive_data is not None:
            selection = alt.selection_point(on="click", encodings=["x", "y"])
        else:
            selection = None

        cm_chart = _plot_cm_chart(df, selection, alt)

        if self.interactive_data is None:
            self.chart = cm_chart
            return self

        sample_data_chart = _plot_sample_data_chart(df, selection, sampled_columns, alt)

        data_statistics_chart = _plot_data_statistics_chart(
            df, selection, metric_columns, alt
        )

        concat_chart = (
            alt.vconcat(cm_chart, sample_data_chart, data_statistics_chart)
            .configure_concat(spacing=100)
            .add_params(selection)
        )

        self.chart = concat_chart
        return self

    def __add__(self, another):
        raise NotImplementedError(
            f"{type(self).__name__!r} doesn't support the add (+) operator"
        )

    def __sub__(self, another):
        raise NotImplementedError(
            f"{type(self).__name__!r} doesn't support the subtract (-) operator"
        )

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.InteractiveConfusionMatrix",
            "cm": self.cm.tolist(),
            "target_names": self.target_names,
            "interactive_data": self.interactive_data,
            "version": __version__,
        }

    @classmethod
    def from_dump(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cm = np.array(data["cm"])
        normalize = data["normalize"]
        target_names = data["target_names"]
        return cls(cm=cm, target_names=target_names, normalize=normalize).plot()

    @classmethod
    @modify_exceptions
    def from_raw_data(
        cls,
        y_true,
        y_pred,
        X_test=None,
        feature_names=None,
        feature_subset=None,
        nsample=5,
        target_names=None,
        normalize=False,
    ):
        """
        Plot confusion matrix.

        .. seealso:: :class:`ConfusionMatrix`

        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Correct target values (ground truth).
        y_pred : array-like, shape = [n_samples]
            Target predicted classes (estimator predictions).
        X_test : array-like, shape = [n_samples, n_features], optional
                Defaults to None. If X_test is passed interactive data
                is displayed upon clicking on each quadrant of the
                confusion matrix.
        feature_names : list of feature names, optional
                feature_names can be passed if X_test passed is a numpy
                array. If not passed, feature names are generated like
                [Feature 0, Feature 1, .. , Feature N]
        feature_subset: list of features, optional
                subset of features to display in the tables. If not passed
                first 5 columns are selected.
        nsample : int, optional
                Defaults to 5. Number of sample observations to display in
                the interactive table if X_test is passed.
        target_names : list
            List containing the names of the target classes. List must be in order
            e.g. ``['Label for class 0', 'Label for class 1']``. If ``None``,
            generic labels will be generated e.g. ``['Class 0', 'Class 1']``
        normalize : bool
            Normalize the confusion matrix

        Examples
        --------

        :doc:`Click here <../classification/cm_interactive>` to see the user guide.

        """

        target_names = _confusion_matrix_validate_predictions(
            y_true, y_pred, target_names
        )
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame):
                X_test = _validate_test_dataset(X_test, feature_names)
            if X_test.isnull().values.any():
                raise ValueError("X_test contains NAN values.")

            if feature_subset:
                X_test = X_test[feature_subset]
            elif len(X_test.columns) > 5:
                warnings.warn(
                    "Only first 5 columns of X_test will be displayed "
                    "on interactive chart"
                )
                X_test = X_test.iloc[:, :5]

        quadrant_indices = _quadrant_wise_indices(y_true, y_pred)
        interactive_data = _quadrant_interactive_data(X_test, quadrant_indices, nsample)
        cm = _confusion_matrix(y_true, y_pred, normalize)

        return cls(
            cm, target_names=target_names, interactive_data=interactive_data
        ).plot()

    @classmethod
    def _from_data(cls, target_names, normalize, cm):
        return cls(cm=np.array(cm), target_names=target_names)
