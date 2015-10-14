#scikit-learn model evaluation

Note: this project is in a very early stage of development.

scikit-learn model evaluation is a package to make model evaluation much easier. 
The package is divided in modules that have defined functionality.

##Plots

Generate evaluation plots with a single function call.
```
from sklearn_model_eval import plots

...code for dataloading and model training...

plots.confusion_matrix_(y_test, y_pred, target_names=target_names)
<img src="tests/baseline_images_confusion_matrix.png">
```



##Tables

Generate good looking tables from your model easily.


##Report generation

Generate HTML reports in one line.


#Design caveats

* Precision-Recall and ROC curve use `label_binarize` each time they are used.

#TODO

* Print lists-like objects as nice html tables in reports
* Include more plots (learning plot, for example)