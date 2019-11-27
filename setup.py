#! /usr/bin/env python
"""
    MICO: Mutual Information and Conic Optimization for feature selection.
"""

import sys
import os
import codecs

from setuptools import setup, find_packages


def load_requirements():
    requirements = [
        "scipy>=1.3.1",
        "numpy>=1.17.3",
        "scikit-learn>=0.21",
        "joblib>=0.14.0",
        "psutil>=5.6.3",
        "pyitlib>=0.2.2"
    ]
    return requirements


def load_version():
    """Executes mico/version.py in a globals dictionary and
    return it.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with codecs.open(os.path.join('mico', 'version.py'),
                     encoding='utf-8-sig') as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'mico'
DESCRIPTION = 'MICO: Mutual Information and Conic Optimization for feature selection.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'KuoLing Huang'
MAINTAINER_EMAIL = 'colinopt.org@gmail.com'
URL = 'https://github.com/glemaitre/mico'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/glemaitre/mico' # To update
VERSION = _VERSION_GLOBALS['__version__']
REQUIREMENTS = load_requirements()


if __name__ == "__main__":

    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_imbalanced_dataset_installing=True)

    install_requires = \
        ['%s>=%s' % (mod, meta['min_version'])
           for mod, meta in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
            if not meta['required_at_installation']]

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: The 3-Clause BSD License',
            # 'Programming Language :: C',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            # 'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        packages=find_packages(),
        install_requires=REQUIREMENTS
    )
