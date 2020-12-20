"""
Tools for comparing model results
"""

# Things to compare:
#
# * Single metrics, all of them can be expressed as a single dictionary, then each experiment is a row and each metric a column
#     * Perform a diff operation (e.g. difference or ratio)
#     * As styled table
#     * Visually
#     * if its just two, add diff column, otherwise display with colors the max and min
# * Parameters, also represented by dicts
#     * But the difference is presented as a list
#     * Like, A, B C parameters are thge same, C is different, here are the values
# * Sliced metrics
#     * Like the first one but each metric is broken down by some other factor
#     * Difference could be represented by the difference of the table but we could only compare two at a time
# * Plots
#     * Add a tab to switch between them

# +
import base64

import papermill as pm
import jupytext
import pandas as pd
from IPython.display import HTML, Image
import matplotlib.pyplot as plt
from jinja2 import Template
# -

nb = jupytext.read('train.py')
# nb.metadata['kernelspec'] = {'name': 'python3', 'language': 'python'}
jupytext.write(nb, 'train.ipynb')

# +
models = ['sklearn.ensemble.RandomForestRegressor', 'sklearn.linear_model.LinearRegression', 'sklearn.svm.LinearSVR']

for model in models:
    pm.execute_notebook('train.ipynb', f'{model}.ipynb', parameters={'model': model})
# -

from sklearn_evaluation import NotebookIntrospector
from sklearn_evaluation.NotebookIntrospector import NotebookCollection

# +
files = [f'{model}.ipynb' for model in models]
ids = ['rf', 'lr', 'svr']

col = NotebookCollection(files, keys=ids)


# +
def html2df(html):
    dfs = pd.read_html(html.data)
    
    if len(dfs) > 1:
        raise ValueError('More than one table detected')
    
    df = dfs[0]

    if df.columns[0] == 'Unnamed: 0':
        df.columns = [None, *df.columns[1:]]

    df = df.set_index(df.columns[0])
    return df

def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def color_neg_and_pos(val):
    color = 'green' if val < 0 else 'red'
    return 'color: %s' % color

def color_max(s):
    is_max = s == s.max()
    return ['color: red' if v else '' for v in is_max]

def color_min(s):
    is_max = s == s.min()
    return ['color: green' if v else '' for v in is_max]

def metrics_df(tables, ids):
    out = pd.concat([html2df(table) for table in tables])
    out.index = ids
    out = out.T
    
    if len(tables) == 2:
        c1, c2 = out.columns
        diff = out[c2] - out[c1]
        # TODO: add ratio and percentage
        out['diff'] = diff
        styled = out.style.applymap(color_neg_and_pos, subset=['diff'])
    else:
        styled = out.style.apply(color_max, axis='columns').apply(color_min, axis='columns')
    
    return styled

def data2html_img(data):
    img = base64.encodebytes(data).decode('utf-8')
    return '<img src="data:image/png;base64, {}"/>'.format(img)

def process_content(content):
    return data2html_img(content.data) if isinstance(content, Image) else content.data


# +
# maybe add a less is better option to color code green/red
m1 = {'mse': 4.0, 'mae': 2.0, 'max_error': 5.5}
m2 = {'mse': 3.5, 'mae': 1.9, 'max_error': 10.1}
m3 = {'mse': 2.5, 'mae': 0.9, 'max_error': 5.1}

# ids = ['exp1', 'exp2', 'exp3']

# +
# out = metrics_df([m1, m2, m3], ids)
# out
# -

d1 = {'mse': {'group1': 4.5, 'group2': 5.5}, 'mae': {'group1': 2.2, 'group2': 1.5}}
d2 = {'mse': {'group1': 5.8, 'group2': 3.2}, 'mae': {'group1': 3.5, 'group2': 2.1}}

df1 = pd.DataFrame(d1).T
df2 = pd.DataFrame(d2).T


def make_tabs(names, contents):
    contents = [process_content(content) for content in contents]
    html = Template("""
<ul class="nav nav-tabs" id="myTab" role="tablist">
  {% for name in names %}
  <li class="nav-item" role="presentation">
    <a class="nav-link" id="{{name}}-tab" data-toggle="tab" href="#{{name}}" role="tab" aria-controls="{{name}}" aria-selected="true">{{name}}</a>
  </li>
  {% endfor %}
</ul>
<div class="tab-content" id="myTabContent">
  {% for name, content in zip(names, contents) %}
  <div class="tab-pane fade" id="{{name}}" role="tabpanel" aria-labelledby="{{name}}-tab">{{content}}</div>
  {% endfor %}
</div>
""").render(names=names, zip=zip, contents=contents)
    return HTML(html)


# +
import copy


def process_elements(elements, ids):
    out = copy.copy(elements)
    out_ids = copy.copy(ids)
    
    if isinstance(elements[0], HTML):
        summary = HTML(metrics_df(elements, ids)._repr_html_())
        out.append(summary)
        out_ids.append('Summary')
    
    return out, out_ids


# -

e, ids_out = process_elements(col['metrics'], ids)
make_tabs(ids_out, e)


