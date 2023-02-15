#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2023, IBM Corp.
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------
"""
Linear regression is a simple but very useful and commonly applied approach
to the regression task, even though it only performs direct modeling of
linear relationships. It is the thing that limits its applicability, a linear
model representation, that makes it fast, efficient, and easy to use (compared
to more refined regression algorithms).
"""
from typing import List
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase

from nzpyida.analytics.predictive.regression import Regression


class LinearRegression(Regression):
    """
    Linear regression predictive model.
    """

    def __init__(self, idadb: IdaDataBase, model_name: str):
        """
        Creates the regressor class.

        Parameters
        ----------

        idada : IdaDataBase
            database connector

        model_name : str
            model name - if it exists in the database, it will be used, otherwise
            it must be trained using fit() function before prediction or scoring is called.
        """

        super().__init__(idadb, model_name)
        self.fit_proc = 'LINEAR_REGRESSION'
        self.predict_proc = 'PREDICT_LINEAR_REGRESSION'


    def fit(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        in_columns: List[str]=None, nominal_colums: str=None, col_def_type: str=None,
        col_def_role: str=None, col_properties_table: str=None, use_svd_solver: bool=False,
        intercept: bool=True, calculate_diagnostics: bool=False):
        """
        Creates a linear regression model based on provided data and store it in a database.

        Parameters
        ----------

        in_df : IdaDataFrame
            the input data frame

        id_column : str, optional
            the input table column identifying a unique instance id

        target_column : str, optional
            the input table column representing the prediction target, definition of multitargets
            can be processed by 'incolumn' parameter and column properties.

        nominal_colums : str, optional
            the input table nominal columns, if any, separated by a semi-colon (;).
            Parameter 'nominalCols' is deprecated please use 'incolumn' intead.

        in_columns : List[str], optional
            the list of input table columns with special properties.
            Each column is followed by one or several of the following properties:
                its type: ':nom' (for nominal), ':cont' (for continuous).
                Per default, all numerical types are con-tinuous, other types are nominal.
                its role: ':id', ':target', ':input', ':ignore'.
            (Remark: ':objweight' is unsupported, i.e. ':objweight' same as ':ignore').
            (Remark: ':colweight(<wgt>)' is unsupported, i.e. ':colweight(<wgt>)' same
            as ':colweight(1)' same as ':input').
            If the parameter is undefined, all columns of the input table have default properties.

        col_def_type : str, optional
            default type of the input table columns. Allowed values are 'nom' and 'cont'.
            If the parameter is undefined, all numeric columns are considered continuous,
            other columns nominal.

        col_def_role : str, optional
            default role of the input table columns. Allowed values are 'input' and 'ignore'.
            If the parameter is undefined, all columns are considered 'input' columns.

        col_properties_table : str, optional
            the input table where column properties for the input table columns are stored.
            The format of this table is the output format of stored procedure
            nza..COLUMN_PROPERTIES().
            If the parameter is undefined, the input table column properties will be
            detected automatically.
            (Remark: colPropertiesTable with "COLROLE" column with value 'objweight' is
            unsupported, i.e. same as 'ignore')
            (Remark: colPropertiesTable with "COLWEIGHT" column with value '<wgt>' is unsup-ported,
            i.e. same as '1')

        use_svd_solver : bool, optional
            a flag indicating whether Singular Value Decomposition and matrix multiplication
            should be used for solving the matrix equation

        intercept : bool, optional
            flag indicating whether the model is built with or without an intercept value.
            The default has changed to true.

        calculate_diagnostics : bool, optional
            a flag indicating whether diagnostics information should be displayed
        """

        params = {
            'id': id_column,
            'target': target_column,
            'nominalCols': nominal_colums,
            'incolumn': in_columns,
            'coldeftype': col_def_type,
            'coldefrole': col_def_role,
            'colpropertiestable': col_properties_table,
            'useSVDSolver': use_svd_solver,
            'intercept': intercept,
            'calculateDiagnostics': calculate_diagnostics
        }

        self._fit(in_df=in_df, params=params)
