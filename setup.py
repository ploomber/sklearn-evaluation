from setuptools import setup

setup(name='sklearn-evaluation',
      packages=['sklearn_evaluation'],
      version='0.2',
      description='Utilities for scikit-learn model evaluation',
      url='http://github.com/edublancas/sklearn-evaluation',
      download_url='https://github.com/edublancas/sklearn-evaluation/tarball/0.2',
      author='Eduardo Blancas Reyes',
      author_email='edu.blancas@gmail.com',
      license='MIT',
      keywords=['datascience', 'machinelearning'],
      classifiers=[],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'decorator'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],  # add freezegun
      )
