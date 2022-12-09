"""
Setup tasks (requires invoke: pip install invoke)
"""
from invoke import task


@task
def setup(c, version=None):
    """
    Setup dev environment, requires conda
    """
    version = version or '3.10'
    suffix = '' if version == '3.10' else version.replace('.', '')
    env_name = f'sk-eval{suffix}'

    c.run(f'conda create --name {env_name} python={version} --yes')
    c.run('eval "$(conda shell.bash hook)" '
          f'&& conda activate {env_name} '
          '&& pip install --editable .[all] '
          '&& pip install invoke lxml'
          '&& pip install jupyter-book'
          '&& pip install ghp-import'
          )  # lxml needed for NotebookCollection.py example
    # jupyter-book and ghp-import added here since it
    # takes too long to resolve dependencies when added
    # in setup.py

    print(f'Done! Activate your environment with:\nconda activate {env_name}')


@task(aliases=['v'])
def version(c):
    """Create a new version
    """
    from pkgmt import versioneer
    versioneer.version(project_root='.', tag=True)


@task(aliases=['r'])
def release(c, tag, production=True):
    """Upload to PyPI
    """
    from pkgmt import versioneer
    versioneer.upload(tag, production=production)
