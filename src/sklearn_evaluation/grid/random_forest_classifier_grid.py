from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn_evaluation import plot
from sklearn_evaluation.grid.classifier_grid import AbstractClassifierGrid, GridTypes


class RandomForestClassifierGrid(AbstractClassifierGrid):

    param_grids = dict({
        GridTypes.TINY: {
            "n_estimators": [1, 10],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
        },
        GridTypes.SMALL: {
            "n_estimators": [1, 10, 50],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
        },
        GridTypes.MEDIUM: {
            "n_estimators": [1, 10, 50, 100],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
        },
        GridTypes.LARGE: {
            "n_estimators": [1, 5],
            "max_features": ["sqrt", "log2"],
            "max_depth": [2, 4],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True],
        },
        GridTypes.X_LARGE: {
            "n_estimators": [1, 10, 50, 100],
            "max_features": ["sqrt", "log2"],
            "max_depth": [2, 4],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True],
        },
    })

    def __init__(self, grid, cv=3, verbose=0):
        super().__init__(grid)
        self.param_grid = self.param_grids[self.grid]
        self.estimator = RandomForestClassifier()
        self.classifier = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=cv,
            verbose=verbose)

    def fit(self, X, y):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        classifier : RandomForestClassifier
            Returns the RandomForestClassifier instance.
        """
        self.X = X
        self.y = y
        self.classifier.fit(X, y, sample_weight=None)
        return self.classifier

    def confusion_matrix(self):
        y_pred = self.classifier.best_estimator_.predict(self.X)
        return plot.confusion_matrix(self.y, y_pred)

    def roc(self):
        y_pred = self.classifier.best_estimator_.predict(self.X)
        return plot.roc(self.y, y_pred)

    def feature_importances(self):
        feature_importances = self.classifier.best_estimator_.feature_importances_
        return plot.feature_importances(feature_importances)

    def grid_search(self, change='n_estimators', kind='line'):
        return plot.grid_search(self.classifier.cv_results_, change=change, kind=kind)
