MICO
====

MICO (Mutual Information and Conic Optimization for feature selection) solves the feature selection problems using mutual information.


Getting Started
---------------

The following steps will walk through how to use MICO. See Sphinx's documentation on
`Getting Started <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


Contributing
------------

Running the tests
~~~~~~~~~~~~~~~~~

Tests are executed through `tox <https://tox.readthedocs.io/en/latest/>`_.

.. code-block:: bash

    tox

Code Style
~~~~~~~~~~

Code is formatted using `black <https://github.com/python/black>`_.

You can check your formatting using black's check mode:

.. code-block:: bash

    tox -e formatting

You can also get black to format your changes for you:

.. code-block:: bash

    black autoapi/ tests/

You can even get black to format changes automatically when you commit using `pre-commit <https://pre-commit.com/>`_:


.. code-block:: bash

    pip install pre-commit
    pre-commit install


Versioning
----------

We use `SemVer <http://semver.org/>`_ for versioning. For the versions available, see the `tags on this repository <https://github.com/readthedocs/sphinx-autoapi/tags>`_.

License
-------

This project is licensed under the MIT License.
See the `LICENSE.rst <LICENSE.rst>`_ file for details.