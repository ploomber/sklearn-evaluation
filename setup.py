from setuptools import setup

setup(name='sklearn-evaluation',
      packages=['sklearn_evaluation'],
      version='0.14',
      description='Utilities for scikit-learn model evaluation (INTERNAL USE ONLY)',
      url='http://github.com/edublancas/sklearn-evaluation',
      download_url = 'https://github.com/edublancas/sklearn-evaluation/tarball/0.14',
      author='Eduardo Blancas Reyes',
      author_email='edu.blancas@gmail.com',
      license='MIT',
      keywords = ['datascience'],
      classifiers = [],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'tabulate',
          #add report generator dependencies
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      )