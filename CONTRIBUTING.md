# Contributing to sklearn-evaluation

For general guidelines, see [Ploombers' documentation framework.](https://github.com/ploomber/contributing/blob/main/documentation/README.md)

## Adding a new plot

To provide a consistent user experience, all the functions that produce plots follow a few conventions:

## Function should not contain `plot` in its name

Example:

```python
def precision_recall(y_true, y_score, ax=None):
    pass
```

## Every argument except the input data should have default values

Example:

```python
def some_algorithm(y_true, y_pred, param=1, another=2, ax=None):
    pass
```

## Follow the argument naming convention

Many of the functions take a vector of real values (first argument) and a vector of predicted values (second argument), they should be named `y_true` and `y_pred` respectively.

```python
def some_algorithm(y_true, y_pred, ..., ax=None):
    pass
```

See the [`confusion_matrix`](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/classification.py#L135) function for an example.

If the plotting function applies to classifiers (e.g., confusion matrix), and the raw scores from the models are the input (instead of the predicted class), the second argument should be named `y_score`:

```python
def some_algorithm(y_true, y_score, ..., ax=None):
    pass
```

See the [`precision_at_proportion`](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/classification.py#L309) function for an example.

In cases where the function doesn't take a true and predicted vector, the names should be descriptive enough:

```python
def some_algorithm(some_meaningful_name, ..., ax=None):
    pass
```

See the [`learning_curve`](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/learning_curve.py#L7) function for an example.

## The last argument in the function must be `ax=None`

The last argument should be `ax=None`. If the user passes a value (a `matplotlib.axes.Axes` object), the plot must be created there. If not, we should use the default axes with `ax = plt.gca()`.

```python
def some_algorithm(a, b, ..., ax=None):
    pass
```

See the [`roc`](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/roc.py#L45) function for an example.

## Functions must return a `matplotlib.Axes` object

The `ax` object must be returned at the end of the function.

See the [`roc`](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/roc.py#L45) function for an example.


## Testing the implementation

Each function must have a corresponding test. If the function has parameters that alter the plot, they should be included as well as separate tests. See the [plot tests](https://github.com/ploomber/sklearn-evaluation/blob/master/tests/test_plot.py) here.

## Documentation

The function must contain a docstring explaining what the function does and a description of each argument. [See this example.](https://github.com/ploomber/sklearn-evaluation/blob/8056bc31ec5e372102d0ee5ada988e380b077c4b/src/sklearn_evaluation/plot/classification.py#L143)

Furthermore, a full example (under the docstring's `Examples` section)must be included in the examples section of the docstring. Such an example must be standalone so that copy-paste should work. [See this example.](https://sklearn-evaluation.readthedocs.io/en/latest/api/plot.html#sklearn_evaluation.plot.confusion_matrix) Note that these examples are automatically tested by the CI.

Each function's docstring should also have a `Notes` section with a `.. versionadded::` to specify from which version this plot is available.
The current dev version of sklearn-evaluation can be found in [`here`](https://github.com/ploomber/sklearn-evaluation/blob/master/src/sklearn_evaluation/__init__.py). So, if current `dev` version is 0.0.1dev, the version of the next release will be 0.0.1.

Here's a docstring template you can use:

```python
def my_plotting_function(y_true, y_pred, ax=None):
    """Plot {plot name}

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).

    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions).

    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/{example-name}.py
    
    Notes
    -----
    .. versionadded:: 0.0.1
    """
    pass
```

## Telemetry : Monitoring the state of `sklearn-evaluation`

Use [`SKLearnEvaluationLogger`](https://github.com/ploomber/sklearn-evaluation/blob/f32c15a43f4a9b4c2e588b3c0f71ba6dc5a71a7e/src/sklearn_evaluation/telemetry.py#L19) decorator to generate logs

Example:

```python
@SKLearnEvaluationLogger.log(feature='plot')
def confusion_matrix(
        y_true,
        y_pred,
        target_names=None,
        normalize=False,
        cmap=None,
        ax=None,
        **kwargs):
pass
```

this will generate the following log:

```json
        {
          "metadata": {
          "action": "confusion_matrix"
          "feature": "plot",
          "args": {
                        "target_names": "None",
                        "normalize": "False",
                        "cmap": "None",
                        "ax": "None"
                    }
          }
        }
```

\*\* since `y_true` and `y_pred` are positional arguments without default values it won't log them

### Queries

1. Run queries and filter out `sklearn-evaluation` events by the event name: `sklearn-evaluation`
2. Break these events by feature ('plot', 'report', 'SQLiteTracker', 'NotebookCollection')
3. Break events by actions/func name (i.e: 'confusion_matrix', 'roc', etc...)

### Errors

Failing runnings will be named: `sklearn-evaluation-error`

## Object-Oriented API

### Adding a new plot

Use [figure](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html) instead of [Figure](https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.figure.Figure.html). When a plot is generated using `Figure`, the `FigureManager` gets destroyed and hence the image becomes inaccessible from example scripts. The solution is to [create a dummy manager](https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure) and use its canvas manager to display the image again. On the other hand, plots generated using `figure` can easily be accessed through `pyplot.show()`. 



