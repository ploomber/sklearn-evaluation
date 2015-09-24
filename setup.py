from setuptools import setup

setup(name='sklearn_model_eval',
      version='0.1',
      description='Dead simple evaluation for scikit-learn models',
      url='http://github.com/',
      author='Eduardo Blancas Reyes',
      author_email='edu.blancas@gmail.com',
      license='MIT',
      packages=['sklearn_model_eval'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False
      )