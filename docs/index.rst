.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to MICO's documentation!
================================


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


.. toctree::
   :maxdepth: 2

   install
   api
   auto_examples/index
   ...

See the `README <https://github.com/thuijskens/stability-selection/blob/master/README.md>`_
for more information.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
