import abc
from enum import Enum


class GridTypes(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    X_LARGE = "xlarge"


class AbstractClassifierGrid(abc.ABC):
    def __init__(self, grid=GridTypes.TINY):
        grid_types = [member.value for member in GridTypes]

        if grid in grid_types:
            self.grid = GridTypes(grid)
        else:
            raise ValueError(
                f"{grid} is not a valid grid value. allowed values : {grid_types} "
            )

    @classmethod
    @abc.abstractmethod
    def set_test_data(self, X_test, y_test):
        pass

    @classmethod
    @abc.abstractmethod
    def fit(self):
        pass

    @classmethod
    @abc.abstractmethod
    def confusion_matrix(self):
        pass

    @classmethod
    @abc.abstractmethod
    def roc(self):
        pass

    @classmethod
    @abc.abstractmethod
    def feature_importances(self):
        pass

    @classmethod
    @abc.abstractmethod
    def grid_search_results(self, change="n_estimators", kind="line"):
        pass
