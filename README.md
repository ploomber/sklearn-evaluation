#scikit-learn model evaluation

Note: this project is in a very early stage of development.

scikit-learn model evaluation is a package to make model evaluation much easier.

#Usage

The package is divided in modules that have defined functionality.

##Plots module

Generate evaluation plots with a single function call.
```python
from sklearn_model_eval import plots

#code for data loading and model training

plots.confusion_matrix_(y_true, y_pred, target_names=target_names)
```

<img src="tests/baseline_images/confusion_matrix.png" style="width:auto;height:auto;max-height:300px">

See this Jupyter [notebook](examples/plots.ipynb) for more examples.

##Tables module

Generate good looking tables from your model easily.

##Report generation module

Generate HTML reports in one line.

#Install

```python
git clone https://github.com/edublancas/sklearn-model-evaluation
cd sklearn-model-evaluation
python setup.py install
```

#Design caveats

* Precision-Recall and ROC curve use `label_binarize` each time they are used.

#TODO

* Print lists-like objects as nice html tables in reports
* Include more plots (learning plot, for example)
* plots.confusion_matrix_ should be plots.confusion_matrix