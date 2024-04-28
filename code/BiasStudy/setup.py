from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='BiasStudy',
   version='1.0',
   packages=['BiasStudy'],
   install_requires=required,
)
