from setuptools import setup, find_packages

exec(open('multilinear_algebra/version.py').read())

setup(
    name='multilinear_algebra',
    version=__version__,
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