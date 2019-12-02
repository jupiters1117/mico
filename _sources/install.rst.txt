Installation
============

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
