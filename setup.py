import setuptools
import re

with open('README.md', 'r') as fh:
    long_description = fh.read()

def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

def get_version():
    VERSION_FILE='pyandev/_version.py'
    version_str = open(VERSION_FILE, 'rt').read()
    VERSION_RE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VERSION_RE, version_str, re.M)
    if mo:
        version = mo.group(1)
    else:
        raise RuntimeError('Unable to find version string in %s.' % (VERSION_FILE,))
    return(version)

setuptools.setup(
    name='pyandev',
    version=get_version(),
    description='Python analysis development package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/richardangell/py-analysis-development',
    author='Richard Angell',
    author_email='richardangell37@gmail.com',
    install_requires=list_reqs(),
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)