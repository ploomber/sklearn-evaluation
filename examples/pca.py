import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn_evaluation import plot

data = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
X = data[0]
y = data[1]

plot.pca(X, y)
plt.show()
