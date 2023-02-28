from setuptools import setup

setup(name='shift_python_utilities',
      version=__version__,
      url='http://github.com/isofit/isofit/',
      author='Evan Lang',
      author_email='evan.d.lang@nasa.gov',
      description='Modules to help make common data operations easier',
      packages=find_packages(),
      install_requires=[],
      python_requires='>=3.7')