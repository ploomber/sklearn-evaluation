import matplotlib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from src.sklearn_evaluation import plot


class TestPlots:

    def setup_method(self):
        # get training and testing data
        X, y = datasets.make_classification(1000,
                                            20,
                                            n_informative=10,
                                            class_sep=0.80,
                                            n_classes=3,
                                            random_state=0)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)

        # fit decision tree and random forest, return confusion matrices
        tree_pred, forest_pred = [
            est.fit(X_train, y_train).predict(X_test)
            for est in [DecisionTreeClassifier(),
                        RandomForestClassifier()]
        ]

        self.tree_cm = plot.ConfusionMatrix.from_raw_data(
            y_test, tree_pred, normalize=False)
        self.forest_cm = plot.ConfusionMatrix.from_raw_data(
            y_test, forest_pred, normalize=False)

    def test_return_val_class(self):
        assert isinstance(self.tree_cm, plot.ConfusionMatrix)

    def test_return_val_figure(self):
        assert isinstance(self.tree_cm.figure, matplotlib.figure.Figure)

    def test_add_plots(self):
        assert isinstance(self.tree_cm + self.forest_cm,
                          matplotlib.figure.Figure)

    def test_subtract_plots(self):
        assert isinstance(self.tree_cm - self.forest_cm,
                          matplotlib.figure.Figure)

    def test_resize_plot_seven(self):
        self.tree_cm.figure.set_size_inches(7, 7)
        size = self.tree_cm.figure.get_size_inches()
        assert size[0] == 7 and size[1] == 7

    def test_resize_plot_nine(self):
        self.tree_cm.figure.set_size_inches(9, 9)
        size = self.tree_cm.figure.get_size_inches()
        assert size[0] == 9 and size[1] == 9
