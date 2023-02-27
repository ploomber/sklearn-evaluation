import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot


# generate data
X, y = datasets.make_classification(
    1000, 5, n_informative=3, class_sep=0.80, n_classes=3, random_state=0
)

# split  data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
columns = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
# plot for classifier 1
df = pd.DataFrame(X_test, columns=columns)
tree_cm = plot.InteractiveConfusionMatrix.from_raw_data(y_test, y_pred, X_test=df)
tree_cm.chart.save("metric_chart.html")
