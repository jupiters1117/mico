.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to MICO's documentation!
================================


MICO: Mutual Information and Conic Optimization for feature selection
---------------------------------------------------------------------

**MICO** is a Python package that implements a conic optimization based feature selection method with mutual information (MI) measure [1]_. The idea behind the approach is to measure the features’relevance and redundancy using MI, and formulate a feature selection problem as a pure-binary quadratic optimization problem, which can be heuristically solved by an efficient randomization algorithm via semidefinite programming [2]_. Optimization software **Colin** [6]_ is used for solving the underlying conic optimization problems.

This package

- implements three methods for feature selections:

  + **MICO** : :ref:`Conic optimization approach for feature selection` (main approach)
  + **MIFS** : :ref:`Backward elimination approach for feature selection`
  + **MIBS** : :ref:`Forward selection approach for feature selection` (less expensive)

- supports three different MI measures:

  + **JMI** : Joint Mutual Information [3]_
  + **JMIM** : Joint Mutual Information Maximisation [4]_
  + **MRMR** : Max-Relevance Min-Redundancy [5]_

- generates feature importance scores for all selected features.
- provides scikit-learn compatible APIs.

Documentation Outline
---------------------

.. toctree::
   :maxdepth: 2

   install
   api
   auto_examples/index

.. See the `README <https://github.com/thuijskens/stability-selection/blob/master/README.md>`_ for more information.


References
----------

.. [1] T Naghibi, S Hoffmann and B Pfister, "A semidefinite programming based search strategy for feature selection with mutual information measure", IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), pp. 1529--1541, 2015. [`Pre-print <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.8516&rep=rep1&type=pdf>`_]
.. [2] M Goemans and D Williamson, "Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming", J. ACM, 42(6), pp. 1115--1145, 1995 [`Pre-print <http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf>`_]
.. [3] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
.. [4] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
.. [5] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
.. [6] Colin: Conic-form Linear Optimizer (`www.colinopt.org <http://www.colinopt.org>`_).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


