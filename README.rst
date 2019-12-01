.. -*- mode: rst -*-
.. template: https://raw.githubusercontent.com/scikit-learn-contrib/stability-selection/master/README.md
.. https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/README.rst

.. image:: https://img.shields.io/pypi/l/colin-mico.svg
    :target: https://github.com/jupiters1117/mico/master/LICENSE
    :alt: License

MICO: Mutual Information and Conic Optimization for feature selection
---------------------------------------------------------------------

**MICO** is a Python package that implements a conic optimization based feature selection method with mutual information (MI) measure [1]_. The idea behind the approach is to measure the features’relevance and redundancy using MI, and then formulate a feature selection problem as a pure-binary quadratic optimization problem, which can be heuristically solved by an efficient randomization algorithm via semidefinite programming [2]_. Optimization software **Colin** [6]_ is used for solving the underlying conic optimization problems.

This package

- implements three methods for feature selections:

  + **MICO** : Conic Optimization approach
  + **MIFS** : Forward Selection approach
  + **MIBS** : Backward Selection approach

- supports three different MI measures:

  + **JMI** : Joint Mutual Information [3]_
  + **JMIM** : Joint Mutual Information Maximisation [4]_
  + **MRMR** : Max-Relevance Min-Redundancy [5]_

- generates feature importance scores for all selected features.
- provides scikit-learn compatible APIs.


Installation
------------

1. Download **Colin** distribution from http://www.colinopt.org/downloads.php and unpack it into a chosen directory (`<CLNHOME>`).
   Then install **Colin** package:

.. code-block:: bash

    cd <CLNHOME>/python
    pip install -r requirements.txt
    python setup.py install

2. Next, install **MICO** package dependencies:

.. code-block:: bash

    pip install -r requirements.txt

3. To install **MICO** package, use:

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


Documentation
-------------

User guide, examples, and API are available `here <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


References
----------

.. [1] T Naghibi, S Hoffmann and B Pfister, "A semidefinite programming based search strategy for feature selection with mutual information measure", IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), pp. 1529--1541, 2015. [`Pre-print <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.8516&rep=rep1&type=pdf>`_]
.. [2] M Goemans and D Williamson, "Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming", J. ACM, 42(6), pp. 1115--1145, 1995 [`Pre-print <http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf>`_]
.. [3] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
.. [4] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
.. [5] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
.. [6] Colin: Conic-form Linear Optimizer (www.colinopt.org).


Credits
-------

- KuoLing Huang, 2019-presents


Licensing
---------

**MICO** is 3-clause BSD licensed.


Note
----

**MICO** is heavily inspired from `MIFS: Parallelized Mutual Information based Feature Selection module <https://github.com/danielhomola/mifs>`_ by Daniel Homola.


