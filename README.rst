.. -*- mode: rst -*-
.. template: https://raw.githubusercontent.com/scikit-learn-contrib/stability-selection/master/README.md


# MICO: Mutual Information and Conic Optimization for feature selection

**MICO** is a Python package that implements a conic optimization based feature selection method with mutual information (MI) measure. The idea behind the approach is to use MI to measure the features’relevance and redundancy, and to formulate the feature selection problem as a pure-binary quadratic optimization problem, which can be heuristically solved by an efficient randomization algorithm via semidefinite programming [2]_. Optimization software Colin [6]_ is used for solving the underlying optimization problems.

This package

- provides scikit-learn compatible APIs.
- implements three different MI measures:

  + **JMI** : Joint Mutual Information [3]_
  + **JMIM** : Joint Mutual Information Maximisation [4]_
  + **MRMR** : Max-Relevance Min-Redundancy [5]_

- implements three methods for feature selections:

  + **MICO** : Conic Optimization approach
  + **MIFS** : Forward selection
  + **MIBS** : Backward selection

- generates feature importance scores for all selected features.


Installation
------------

1. Download the 64bit x86 Colin distribution from http://www.colinopt.org/downloads.php and unpack it into a chosen directory. Install Colin-Python package

.. code-block:: bash

    cd <CLNHOME>/python
    pip install -r requirements.txt
    python setup.py install

.. Note::

    User must replace `<CLNHOME>` with the name of your Colin installation directory.

2. Install package dependencies.

.. code-block:: bash

    pip install -r requirements.txt

3. To execute the package, execute:

.. code-block:: bash

    python setup.py install

or

.. code-block:: bash

    pip install colin-mico

To install the development version, you may use:

.. code-block:: bash

    pip install --upgrade git+https://github.com/jupiters1117/mico


Usage
-----

This package provides scikit-learn compatible APIs:

* ``fit(X, y)``
* ``transform(X)``
* ``fit_transform(X, y)``


Examples
--------

The following example illustrates the use of the package:

.. code-block:: python

    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    # Prepare data.
    data = load_breast_cancer()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Perform feature selection.
    mico = MutualInformationConicOptimization(verbose=1, categorical=True)
    mico.fit(X, y)

    # Populate selected features.
    print("Selected features: {}".format(mico.get_support()))

    # Populate feature importance scores.
    print("Feature importance scores: {}".format(mico.feature_importances_))

    # Call transform() on X.
    X_transformed = mico.transform(X)


Getting Started
---------------

The following steps will walk through how to use MICO. See Sphinx's documentation on
`Getting Started <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


References
----------

.. [0] T. Naghibi, S. Hoffmann and B. Pfister, "A Semidefinite Programming Based Search Strategy for Feature Selection with Mutual Information Measure", IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), pp. 1529--1541, 2015. [`Pre-print <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.8516&rep=rep1&type=pdf>`_]
.. [1] T. Naghibi, S. Hoffmann and B. Pfister, "A semidefinite programming based search strategy for feature selection with mutual information measure", IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), pp. 1529--1541, 2015. [`Pre-print <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.8516&rep=rep1&type=pdf>`_]
.. [2] M.X. Goemans and D.P. Williamson, "Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming", J. ACM, 42(6), pp. 1115--1145, 1995 [`Pre-print <http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf>`_]
.. [3] H.H. Yang and J. Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
.. [4] M. Bennasar, Y. Hicks, abd R. Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
.. [5] H. Peng, F. Long, C. Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
.. [6] `Colin: Conic-form Linear Optimizer <www.coliopt.org>`_


Authors
-------

- KuoLing Huang, 2019-presents

