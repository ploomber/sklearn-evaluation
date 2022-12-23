Functional vs OOP
=================

Why having two different ways of doing the same? While the plot/table module
can be accessed directly or via :class:`sklearn_evaluation.ClassifierEvaluator`, they serve slighly
different purposes.

The purpose of :class:`sklearn_evaluation.ClassifierEvaluator` is to provide a simpler API where you can
quickly plot and evaluate a model(s) and generate reports from them
(right now the only way of generating reports is to use the OOP interface).
Since the OOP is simpler, it also has some constraints. When
plotting a confusion matrix from the plot module, you can pass a
:class:`matplotlib.axes.Axes` object which gives you great flexibility,
you can use this to plot a 2 x 2 grid with 4 confusion matrices for different models for example, or to customize the style and elements in the plot.
