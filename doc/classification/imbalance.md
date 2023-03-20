---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
html_meta:
  "description lang=en": "Learn how to evaluate class imbalance in datasets using Target Analysis with the sklearn-evaluation library in Python. Understand the importance of addressing class imbalance in machine learning and visualize class distributions using bar charts."
  "keywords": "class imbalance, target analysis, sklearn-evaluation, Python, machine learning, dataset, resampling techniques, under-sampling, over-sampling, bar chart, majority class, minority class"
  "property=og:locale": "en_US"
---

# Evaluating class imbalance

Class imbalance occurs when the distribution of data points across the known classes are skewed. It's a common problem in machine learning and can affect the model accuracy. Standard classification algorithms work well for a fairly balanced dataset, however when the data is imbalanced the model tends to learn more features from the majority classes as compared to minority classes. 

One common approach of solving this problem is to either decrease the number of samples in the majority class (under-sampling) or increase the number of samples in the minority class(over-sampling). 

It's essential to understand the class imbalance before implementing any resampling techniques. Target analysis helps to visualise the class imbalance in the dataset by creating a bar chart of the frequency of occurence of samples across classes in the dataset

```{code-cell} ipython3
import matplotlib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot
```

```{code-cell} ipython3
matplotlib.rcParams["figure.figsize"] = (7, 7)
matplotlib.rcParams["font.size"] = 18
```

```{code-cell} ipython3
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_classes=2,
    # Set label 0 for  97% and 1 for rest 3% of observations
    weights=[0.85],
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

##### Balance Mode

When only training data is passed the balance mode is displayed which shows distribution of each class.

In the below example we can see that class 0 is the dominating class, hence classifier may have a bias towards this class and predict class 0 most of the time.

```{code-cell} ipython3
plot.target_analysis(y_train)
```

##### Compare Mode

When both the training and the test sets are passed, a side by side bar chart of both the sets is displayed.

The below chart shows that distribution of samples is fairly similar across the train and test splits.

```{code-cell} ipython3
plot.target_analysis(y_train, y_test)
```
