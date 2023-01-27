import os
import platform

import nox


# NOTE: adding
# venv_params=["--channel", "conda-forge"]
# is breaking the installation
@nox.session(
    venv_backend="conda",
    python=os.environ.get("PYTHON_VERSION", "3.8"),
)
def tests(session):
    session.install("pip", "--upgrade")
    session.install("pytest-timeout")

    # if we remove the --editable flag pytest throws an error, because there
    # are two copies of the pkg (src/ and site-packages/), this is a quick
    # way to fix it
    # https://github.com/pytest-dev/pytest/issues/7678
    session.install("--editable", ".")

    # test vanilla installation is importable
    session.run("python", "-c", "import sklearn_evaluation")

    # this is needed to run tests
    session.conda_install("lxml")
    # install other test dependencies
    session.install("--editable", ".[all]")

    # run unit tests
    # docstrings
    # pytest doctest docs: https://docs.pytest.org/en/latest/doctest.html
    # doctest docs: https://docs.python.org/3/library/doctest.html
    # and examples (this is a hacky way to do it since --doctest-modules will
    # first load any .py files, which are the examples, and then try to run
    # any doctests, there isn't any)
    session.run(
        "pytest",
        "tests/",
        "src/",
        "examples/",
        "--cov=sklearn_evaluation",
        "--doctest-modules",
        "--verbose",
        "--timeout=30",
    )
    session.run("coveralls")

    if session.python == "3.8" and platform.system() == "Linux":
        # build docs so we can detect build errors
        session.run("jupyter-book", "build", "docs/", external=True)
