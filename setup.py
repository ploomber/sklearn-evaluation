import re
import ast
from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('sklearn_evaluation/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

DOWNLOAD_URL = ('https://github.com/edublancas/sklearn-evaluation/tarball/{}'
                .format(VERSION))

setup(name='sklearn-evaluation',
      packages=find_packages(exclude=['tests']),
      version=VERSION,
      description=('scikit-learn model evaluation made easy: plots, tables and'
                   'markdown reports.'),
      url='http://github.com/edublancas/sklearn-evaluation',
      download_url=DOWNLOAD_URL,
      author='Eduardo Blancas Reyes',
      author_email='edu.blancas@gmail.com',
      license='MIT',
      keywords=['datascience', 'machinelearning'],
      classifiers=[],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'six',
          'decorator'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],  # add freezegun
      )
