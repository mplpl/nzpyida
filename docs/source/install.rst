Install
*******

Installing
----------

Install nzpyida from pip by issuing this statement::

	> pip install nzpyida

Updating
--------

To update your nzpyida installation, issue this statement::

	> pip install nzpyida --update --no-deps

Ensure you have INZA version 11.2.1.0 or later installed in your Netezza database.

Introduction
------------

This statement builds the project from source::

 	> python setup.py install

These statements build the documentation::

	> cd docs
	> make html
	> make latex

These statements run the test::

	> cd tests
	> py.test

By default, pytest assumes that a database named "BLUDB" is reachable via an ODBC connection and that its credentials are stored in your ODBC settings. This may be not the case for most users, so several options are avaiable:

	* ``--dsn`` : Data Source Name
	* ``--uid``, ``--pwd`` : Database login and password
	* ``--jdbc`` : jdbc url string
	* ``--table`` : Table to use to test (default: iris)

For testing, all tables from nzpyida.sampledata can be used. These include the tables iris, swiss, and titanic.

Strict dependencies
-------------------

nzpyida uses data structures and methods from two common Python libraries - Pandas and Numpy - and additionally depends on a few pure-python libraries, which makes nzpyida easy to install:

	* pandas
	* numpy
	* future
	* six
	* lazy

In addition, for establishing connection to Netezza database one of the following libraries is needed:

	* nzpy - for nzpy connectivity
	* pypyodbc - for ODBC connectivity
	* jaydebeapi - for JDBC connectivity

Optional dependencies
---------------------

Some optional libraries can be installed to benefit from extra features, for example:

	* pytest (for running tests)
	* sphinx (for building the documentation)
