"""
Calibration curve

NOTE: this is largely based in the scikit-plot implementation. License below.

MIT License

Copyright (c) [2018] [Reiichiro Nakano]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve as sk_calibration_curve


# TODO: add unit tests
def calibration_curve(y_true,
                      probabilities,
                      clf_names=None,
                      n_bins=10,
                      cmap='nipy_spectral',
                      ax=None):
    """
    Plots calibration curves for a set of classifier probability estimates.
    Calibration curves help determining whether you can interpret predicted
    probabilities as confidence level. For example, if we take a
    well-calibrated and take the instances where the score is 0.8, 80% of those
    instanes should be from the positive class. This function only works for
    binary classifiers.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]:
        Ground truth (correct) target values.

    probabilities : list of array-like, shape (n_samples, 2) or (n_samples,)
        A list containing the outputs of binary classifiers'
        :func:`predict_proba` method or :func:`decision_function` method.

    clf_names : list of str, optional)
        A list of strings, where each string refers to the name of the 
        classifier that produced the corresponding probability estimates in
        `probabilities`. If ``None``, the names "Classifier 1", "Classifier 2",
        etc. will be used.

    n_bins : int, optional, default=10
        Number of bins. A bigger number requires more data.

    cmap : string or :class:`matplotlib.colors.Colormap` instance, optional
        Colormap used for plotting the projection. View Matplotlib Colormap
        documentation for available options.
        https://matplotlib.org/users/colormaps.html

    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/calibration_curve.py
    """
    y_true = np.asarray(y_true)

    if not isinstance(probabilities, list):
        raise ValueError('`probabilities` does not contain a list.')

    classes = np.unique(y_true)

    if len(classes) > 2:
        raise ValueError('plot_calibration_curve only '
                         'works for binary classification')

    if clf_names is None:
        clf_names = [
            'Classifier {}'.format(x + 1) for x in range(len(probabilities))
        ]

    if len(clf_names) != len(probabilities):
        raise ValueError('Length {} of `clf_names` does not match length {} of'
                         ' `probabilities`'.format(len(clf_names),
                                                   len(probabilities)))

    if ax is None:
        ax = plt.gca()

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i, probas in enumerate(probabilities):
        probas = np.asarray(probas)
        if probas.ndim > 2:
            raise ValueError('Index {} in probabilities has invalid '
                             'shape {}'.format(i, probas.shape))
        if probas.ndim == 2:
            probas = probas[:, 1]

        if probas.shape != y_true.shape:
            raise ValueError('Index {} in probabilities has invalid '
                             'shape {}'.format(i, probas.shape))

        probas = (probas - probas.min()) / (probas.max() - probas.min())

        (fraction_of_positives,
         mean_predicted_value) = sk_calibration_curve(y_true,
                                                      probas,
                                                      n_bins=n_bins)

        color = plt.cm.get_cmap(cmap)(float(i) / len(probabilities))

        ax.plot(mean_predicted_value,
                fraction_of_positives,
                's-',
                label=clf_names[i],
                color=color)

    ax.set_title('Calibration plots (Reliability Curves)')
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')

    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')

    return ax
