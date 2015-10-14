#scikit-learn model evaluation

**Note: this project is in a very early stage of development.**

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

```python
from sklearn_model_eval.report import ReportGenerator

#code for data loading and model training

#Instantiate ReportGenerator with a path to save the HTML files
rg = ReportGenerator(savepath='~/my-awesome-models')
#Generate report
rg.generate_report(model=forest, y_true=y_true, y_pred=y_pred, y_score=y_score, feature_list=feature_list, target_names=target_names, name=name)
```

The code above will generate a report [like this.](http://htmlpreview.github.com/?https://github.com/edublancas/sklearn-model-evaluation/blob/master/examples/sample_report.html)

Reports are self-contained, all images are included in the html file using [base64](https://en.wikipedia.org/wiki/Base64).

#Install

```bash
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