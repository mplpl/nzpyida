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
In many classification applications it may be required or desirable not
only to accurately classify instances, but also to inspect the model.
The inspection makes it possible to explain its decisions, modify it,
or combine with some existing background knowledge. In such applications,
where both the high classification accuracy and human-readability of the
model are required, the method of choice is typically going to be
decision trees.

A decision tree is a hierarchical structure that represents a classification
model using a "divide and conquer" approach. Internal tree nodes represent
splits applied to decompose the data set into subsets, and terminal nodes,
also referred to as leaves, assign class labels to sufficiently small
or uniform subsets. Splits are specified by logical conditions based on
selected single attributes, with a separate outgoing branch corresponding
to each possible outcome.

The concept of decision tree construction is to select splits that decrease
the impurity of class distribution in the resulting subsets of instances,
and increase the domination of one or more classes over the others.
The goal is to find a subset containing only or mostly instances of one
class after a small number of splits, so that a leaf with that class
label is created. This approach promotes simple trees, which typically
generalize better.
"""

from typing import List
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.predictive.classification import Classification


class DecisionTreeClassifier(Classification):
    """
    Decision tree based classifier.
    """

    def __init__(self, idadb: IdaDataBase, model_name: str):
        """
        Creates the classifier class.

        Parameters
        ----------

        idada : IdaDataBase
            database connector

        model_name : str
            model name - if it exists in the database, it will be used, otherwise
            it must be trained using fit() function before prediction or scoring is called.
        """

        super().__init__(idadb, model_name)
        self.fit_proc = 'DECTREE'
        self.predict_proc = 'PREDICT_DECTREE'

    def fit(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        in_columns: List[str]=None, col_def_type: str=None, col_def_role: str=None,
        col_properties_table: str=None, weights: str=None, eval_measure: str=None,
        min_improve: float=0.02, min_split: int=50, max_depth: int=10, val_table: str=None,
        val_weights: str=None, qmeasure: str=None, statistics: str=None):
        """
        Grows the decision tree and stores its model in the database.

        Parameters
        ----------

        in_df : IdaDataFrame
            the input data frame

        id_column : str, optional
            the input table column identifying a unique instance id

        target_column : str, optional
            the input table column representing the class

        in_columns : str, optional
            the list of input table columns with special properties.
            Each column is followed by one or several of the following properties:
                its type: ':nom' (for nominal), ':cont' (for continuous).
                    Per default, all numerical types are continuous, other types are nominal.
                its role: ':id', ':target', ':input', ':ignore'.
            (Remark: ':objweight' is unsupported, i.e. ':objweight' same as ':ignore').
            (Remark: ':colweight(<wgt>)' is unsupported, i.e. ':colweight(<wgt>)' same as
            ':col-weight(1)' same as ':input').
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
            The format of this table is the output format of stored procedure nza..COLUMN_PROPERTIES().
            If the parameter is undefined, the input table column properties will be detected automatically.
            (Remark: colPropertiesTable with "COLROLE" column with value 'objweight' is unsupported,
            i.e. same as 'ignore')
            (Remark: colPropertiesTable with "COLWEIGHT" column with value '<wgt>' is unsupported,
            i.e. same as '1')

        weights : str, optional
            the input table containing optional instance or class weights for the input table columns.
            If the parameter is undefined, we assume that the weights are uniformly equal to 1.
            The <weights> table contains following columns:
                weight: a numeric column containing the instance or class weight,
                id: a column to be joined with the <id> column of <intable>, defining instance weights,
                class: a column to be joined with the <target> column of <intable>, defining class weights.
            The id or class column can be missing, at least one of them must be present.
            For instances or classes not occurring in this table, weights of 1 are assumed.

        eval_measure : str, optional
            the class impurity measure used for split evaluation. Allowed values are 'entropy' and 'gini'

        min_improve : float, optional
            the minimum improvement of the split evaluation measure required

        min_split : int, optional
            the minimum number of instances per tree node that can be split

        max_depth : int, optional
            the maximum number of tree levels (including leaves)

        val_table : str, optional
            the input table containing the validation dataset.
            If this parameter is undefined, no pruning will be performed.

        val_weights : str, optional
            the input table containing optional instance or class weights for the validation dataset.
            It is similar to the <weights> table.

        qmeasure : str, optional
            the quality measure for pruning.
            Allowed values are Acc or wAcc.

        statistics : str, optional
            flags indicating which statistics to collect.
            Allowed values are: none, columns, values:n, all.
            If statistics=none, no statistics are collected.
            If statistics=columns, statistics on the input table columns like mean value are collected.
            If statistics=values:n with n a positive number, statistics about the columns and the column
            values are collected. Up to <n> column value statistics are collected:
                If a nominal column contains more than <n> values, only the <n> most frequent column
                statistics are kept.
                If a numeric column contains more than <n> values, the values will be discretized and
                the statistics will be collected on the discretized values.
            Indicating statistics=all is equal to statistics=values:100.
        """
        
        params = {
            'id': id_column,
            'target': target_column,
            'incolumn': in_columns,
            'coldeftype': col_def_type,
            'coldefrole': col_def_role,
            'colpropertiestable': col_properties_table,
            'weights': weights,
            'eval': eval_measure,
            'minimprove': min_improve,
            'minsplit': min_split,
            'maxdepth': max_depth,
            'valtable': val_table,
            'valweights': val_weights,
            'qmeasure': qmeasure,
            'statistics': statistics
        }

        self._fit(in_df=in_df, params=params)

    def predict(self, in_df: IdaDataFrame, out_table: str=None, id_column: str=None,
        target_column: str=None, prob: bool=False, out_table_prob: str=None) -> IdaDataFrame:
        """
        Makes predictions based on this model. The model must exist.

        Parameters
        ----------

        in_df : IdaDataFrame
            the input data frame

        out_table : str, optional
            the output table where the predictions will be stored

        id_column : str, optional
            the input table column identifying a unique instance id

        target_column : str, optional
            the input table column representing the class

        prob : bool, optional
            the flag indicating whether the probability of the predicted class should be included
            into the output table or not

        out_table_prob : str, optional
            if specified, the probability output table where class probability predictions
            will be stored

        Returns
        -------
        IdaDataFrame
            the data frame containing row identifiers and predicted target values
        """

        params = {
            'id': id_column,
            'target': target_column,
            'prob': prob,
            'outtableprob': out_table_prob
        }

        return self._predict(in_df=in_df, params=params, out_table=out_table)
