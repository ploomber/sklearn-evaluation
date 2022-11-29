# Building docs

Install requirements (this requires conda):
cd to sklearn-evaluation directory and run
```
pip install invoke
invoke setup
```

Install `pandoc`
```
pip install pandoc
```
Build docs:

```
cd docs
make <format>
```

For available formats run `make help`

If changes are made to docs, run `pip install ".[dev]"` before `make` to preview latest changes.