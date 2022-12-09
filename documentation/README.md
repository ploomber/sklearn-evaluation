# Building docs

Install requirements (this requires conda):
cd to sklearn-evaluation directory and run
```
pip install invoke
invoke setup
```

Build docs locally:

```
jupyter-book build documentation/
```

To ensure a clean build:

``` 
jupyter-book clean documentation --all
jupyter-book build documentation/
```

Follow this [guide](https://jupyterbook.org/en/stable/publish/gh-pages.html#use-a-custom-domain-with-github-pages) for publishing the doc online.

If changes are made to docs, run `pip install ".[dev]"` before `build` to preview latest changes.