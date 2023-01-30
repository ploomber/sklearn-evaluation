# Building docs

Install requirements (this requires conda):
cd to sklearn-evaluation directory and run
```
pip install invoke
invoke setup
```

Build docs locally:

```
cd docs
python -m sphinx -T -E -W --keep-going -b html -d _build/doctrees -D language=en . _build/html
```

To ensure a clean build:

``` 
jupyter-book clean docs/ --all
cd docs
python -m sphinx -T -E -W --keep-going -b html -d _build/doctrees -D language=en . _build/html
```

If changes are made to docs, run `pip install ".[dev]"` before `build` to preview latest changes.