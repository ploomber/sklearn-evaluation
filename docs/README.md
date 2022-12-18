# Building docs

Install requirements (this requires conda):
cd to sklearn-evaluation directory and run
```
pip install invoke
invoke setup
```

Build docs locally:

```
jupyter-book build docs/
```

To ensure a clean build:

``` 
jupyter-book clean docs/ --all
jupyter-book build docs/
```

Follow this [guide](https://jupyterbook.org/en/stable/publish/gh-pages.html#use-a-custom-domain-with-github-pages) for publishing the doc online.

If changes are made to docs, run `pip install ".[dev]"` before `build` to preview latest changes.