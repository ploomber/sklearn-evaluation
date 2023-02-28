from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn_evaluation import grid

# generate data
X, y = make_classification(
    n_samples=20000, n_features=2, n_informative=2, n_redundant=0, random_state=0
)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)
model = grid.RandomForestClassifierGrid(grid="tiny")
model.fit(X_train, y_train)
model.set_test_data(X_test, y_test)
model.roc()
