from jinja2 import Environment, PackageLoader


def jinja_env():
    env = Environment(
        loader=PackageLoader('sklearn_evaluation', 'assets/report'),
    )
    return env
