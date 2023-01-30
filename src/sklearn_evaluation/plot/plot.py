import json
from pathlib import Path
import abc

from sklearn_evaluation.report.serialize import figure2html


class AbstractPlot(abc.ABC):
    """An abstract class for all class-based plots"""

    @abc.abstractmethod
    def __init__(self, *, label=None):
        """
        The constructor must take all input data required to create the plot and assign
        it as attributes. e.g., ``self.value = value``, no other processing should
        happen here. The input data must be already aggregated. For example, when
        plotting a confusion matrix, the aggregated data are the numbers that appear
        in each of the quadrants (the unagreggated data are the ``y_true``, ``y_pred``
        arrays). Since users typically create plots from raw data (such as ``y_true``
        and ``y_pred``), they will often use ``Plot.from_raw_data`` instead of this
        constructor.

        The only suggested argument is ``label=None``, which should be used to
        identify the plot (e.g., in the title), and in composed plots.

        All arguments beyond the input data must be keyword-only (add a *
        argument between the input and the rest of the arguments).
        """
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """
        All plotting related code must be here with one optional argument ``ax=None``.
        Must assign, ``self.ax_``, and ``self.figure_`` attributes and return ``self``.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_raw_data(cls):
        """Takes raw unaggregated (for an example of aggregated vs unaggregated data
        see the constructor docstring) data, compute statistics and initializes the
        object. This is the method that users typically use. (e.g., they pass
        ``y_true``, and ``y_pred`` here, we aggregate and call the constructor).

        Apart from input data, this method must have the same argument as the
        constructor.

        All arguments beyond the input data must be keyword-only (add a *
        argument between the input and the rest of the arguments).
        """
        pass

    @abc.abstractmethod
    def _get_data(self):
        """Must return a dictionary with the data required to serialize the object,
        this used by ``AbstractPlot.dump``. The dictionary should contain a key
        "class" with the plot's dotted path (e.g.,
        ``sklearn_evaluation.plot.SomePlot``), and a "version" key with the current
        package version.
        """
        pass

    def __add__(self, another):
        """Optional method to support the ``a + b`` operation. must return an
        ``AbstractComposedPlot`` instance.  This should produce composed plot that
        compares this plot and ``another`` plot.
        """
        raise NotImplementedError(
            f"{type(self).__name__!r} doesn't support the substract (-) operator"
        )

    def __sub__(self, another):
        """Optional method to support the ``a - b`` operation. must return an
        ``AbstractComposedPlot`` instance. This should produce composed plot that
        compares the difference between this plot and ``another`` plot.
        """
        raise NotImplementedError(
            f"{type(self).__name__!r} doesn't support the add (+) operator"
        )

    @classmethod
    def _from_data(cls):
        """
        Optional method to initialize the plot from a dictionary produced by
        ``AbstractPlot.dump``, used for integrating the plot with the experiment
        tracker.
        """
        raise NotImplementedError(f"{cls.__name__!r} doesn't support _from_data")

    def dump(self, path):
        """Serialize the plot as ``.json`` to the given path."""
        data = self._get_data()
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def from_dump(cls, path):
        """Instantiates a plot object from a path to a JSON file. A default
        implementation is provided, but you might override it.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        del data["version"]
        del data["class"]
        return cls(**data).plot()

    def to_html(self):
        return figure2html(self.figure_)


class AbstractComposedPlot(abc.ABC):
    @abc.abstractmethod
    def plot(self, ax=None):
        """
        All plotting related code must be here with one optional argument ``ax=None``.
        Must return ``self``.
        """
        pass
