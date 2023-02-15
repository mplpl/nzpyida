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
Regression trees are decision trees adapted to the regression task, which store
numeric target attribute values instead of class labels in leaves, and use
appropriately modified split selection and stop criteria.

As with decision trees, regression tree nodes decompose the data into subsets,
and regression tree leaves correspond to sufficiently small or sufficiently
uniform subsets. Splits are selected to decrease the dispersion of target
attribute values, so that they can be reasonably well predicted by their mean
values at leaves. The resulting model is piecewise-constant, with fixed
predicted values assigned to regions to which the domain is decomposed by
the tree structure.
"""
from typing import List
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.predictive.regression import Regression
from nzpyida.analytics.utils import map_to_props

class DecisionTreeRegressor(Regression):
    """
    Decision tree based regressor
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
        self.fit_proc = 'REGTREE'
        self.predict_proc = 'PREDICT_REGTREE'
        self.target_column_in_output = "CLASS"
        self.id_column_in_output = 'ID'


    def fit(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        in_columns: List[str]=None, col_def_type: str=None, col_def_role: str=None,
        col_properties_table: str=None, eval_measure: str=None, min_improve: float=0.1,
        min_split: int=50, max_depth: int=10, val_table: str=None, qmeasure: str=None,
        statistics: str=None):
        """
        This function creates a regression tree model based on provided data and
        store it in a database.

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

        eval_measure : str, optional
            the split evaluation measure. Allowed values are: variance.

        min_improve : float, optional
            the minimum improvement of the split evaluation measure required

        min_split : int, optional
            the minimum number of instances per tree node that can be split

        max_depth : int, optional
            the maximum number of tree levels (including leaves)

        val_table : str, optional
            the input table containing the validation dataset.
            If this parameter is undefined, no pruning will be performed.

        qmeasure : str, optional
            the quality measure for pruning the tree. Allowed values are: mse, r2.

        statistics : str, optional
            flags indicating which statistics to collect.
            Allowed values are: none, columns, values:n, all.
            If statistics=none, no statistics are collected.
            If statistics=columns, statistics on the input table columns like mean value
            are collected.
            If statistics=values:n with n a positive number, statistics about the columns
            and the column val-ues are collected. Up to <n> column value statistics
            are collected:
            If a nominal column contains more than <n> values, only the <n> most frequent
            column stat-istics are kept.
            If a numeric column contains more than <n> values, the values will be discretized
            and the stat-istics will be collected on the discretized values.
            Indicating statistics=all is equal to statistics=values:100.

        """

        params = {
            'id': id_column,
            'target': target_column,
            'incolumn': in_columns,
            'coldeftype': col_def_type,
            'coldefrole': col_def_role,
            'colpropertiestable': col_properties_table,
            'eval': eval_measure,
            'minimprove': min_improve,
            'minsplit': min_split,
            'maxdepth': max_depth,
            'valtable': val_table,
            'qmeasure': qmeasure,
            'statistics': statistics
        }

        self._fit(in_df=in_df, params=params)

    def predict(self, in_df: IdaDataFrame, out_table: str=None, id_column: str=None, 
        target_column: str=None, variance: bool=False):
        """
        Makes predictions based on this model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame to predict

        out_table : str, optional
            the output table where the predictions will be stored

        id_column : str, optional
            the input table column identifying a unique instance id
            Default: <id> column used to build the model

        target_column : str, optional
            the input table column representing the prediction target.
            The specified target column will not be used for prediction and is permitted
            to contain NULL values.
            Default: <target> column used to build the model.

        variance : bool, optional
            a flag indicating whether the variance of the predictions should be included
            into the output table

        Returns
        -------
        IdaDataFrame
            the data frame containing row identifiers and predicted target values
        """

        params = {
            'id': id_column,
            'target': target_column,
            'var': variance
            }

        return self._predict(in_df=in_df, params=params, out_table=out_table)

    def __str__(self):
        params = map_to_props({'model': self.model_name})
        return self.idadb.ida_query(f'call NZA..PRINT_REGTREE(\'{params}\')')[0]
