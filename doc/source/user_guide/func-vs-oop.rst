Functional vs Object Oriented interface
=======================================

Why having two different ways of doing the same? While the plot/table module
can be accessed directly from their modules or via ClassifierEvaluator, they
serve slighly different purposes.

The purpose of ClassifierEvaluator is to prove a simpler API where you can
quickly plot an evaluare a model (or several) and generate repors from them
(right now the only way of generating reports is to use the OOP interface).
Since the OOP is simpler, it also has some constraints. When
plotting a confusion matrix from the plot module, you can pass a matplotlib
Axes object which gives you great flexibility, you can use this to plot a 2 x 2
grid with 4 confusion matrices for different models for example, or to
customize the style and elements in the plot.
