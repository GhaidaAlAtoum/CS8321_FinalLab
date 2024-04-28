from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='BiasStudy',
   version='1.0',
   packages=find_packages(),
   install_requires=required,
)
