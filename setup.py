from setuptools import setup, find_packages

setup(
    name='multilinear_algebra',
    version='1.0.0',
    packages=find_packages(),
    author='Andreas Himmel',
    author_email='andreas.himmel@iat.tu-darmstadt.de',
    url='',
    license='GNU Lesser General Public License version 3',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "tabulate",
        "numpy",
        "casadi",
    ],
)