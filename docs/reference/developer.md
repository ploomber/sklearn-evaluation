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
---

# Developer guide

```{note}
Note that this guide will only cover the basics. For more details, see the `AbstractPlot` (abstract class), and `MyBar` (example class) implementations, which contain more information.
```

This guide will show you how to add new plots to `sklearn-evaluation`.

## Object-oriented API

All plots are implemented using an object-oriented API, this implies that each plot is a class.

Users interact with the plots via a `from_raw_data` method, that takes raw, unaggregated data as input, optinal arguments to customize the plot, and an optional `name` argument to idenfity the plot.

Let's see an example. Our example plot takes a list of elements and it produces a bar plot with the count for each different value:

```{code-cell} ipython3
from sklearn_evaluation.plot._example import MyBar

bar = MyBar.from_raw_data(["banana", "banana", "apple", "pineapple", "apple"],
                           color="lightblue",
                           name="first experiment")
```

The constructor must take aggregated data. End-users do not typically call the constructor directly; however, it's still part of the public API. Unlike our `from_raw_data`, the constructor takes aggregated data (the counts):

```{code-cell} ipython3
bar_second = MyBar({"banana": 3, "apple": 5, "pineapple": 2})
```

Plots implement a `.plot()` method where all the plotting logic happens:

```{code-cell} ipython3
bar_second.plot()
```

### Addition

The plot might support the `+` operator, which should produce a combined plot to compare both plots (order might matter in some cases, but not always):

```{code-cell} ipython3
bar + bar_second
```

### Substraction

If it makes sense for your plot, you might add support for the `-` operator, which should create a combined plot to summarize results from two in individual plots:

```{code-cell} ipython3
bar_second - bar
```

Note that order is important:

```{code-cell} ipython3
bar - bar_second
```

### Serialization

Plots should implement a private `_get_data()` method to return all necessary data required to re-create the plot:

```{code-cell} ipython3
bar._get_data()
```

A `.dump()` method to dump this data into a JSON file:

```{code-cell} ipython3
bar.dump("bar.json")
```

And a `from_dump()` class method to load a plot from a JSON file:

```{code-cell} ipython3
MyBar.from_dump("bar.json")
```

## Functional API

```{code-cell} ipython3
from sklearn_evaluation.plot._example import my_bar

result = my_bar(["banana", "banana", "apple", "pineapple", "apple"],
                color="lightblue")
```

```{code-cell} ipython3
type(result)
```

## General guidelines

```{code-cell} ipython3

```
