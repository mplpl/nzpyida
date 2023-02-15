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
The naive Bayes classifier is a simpler classification algorithm than most,
which makes it quick and easy to apply. While it does not compete with more
sophisticated algorithms with respect to classification accuracy, in some
cases it may be able to deliver similar results in a fraction of the
computation time.
"""
from typing import List
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.predictive.classification import Classification

class NaiveBayesClassifier(Classification):
    """
    Naive Bayes classifier
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
        self.fit_proc = 'NAIVEBAYES'
        self.predict_proc = 'PREDICT_NAIVEBAYES'

    def fit(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        in_columns: List[str]=None, col_def_type: str=None, col_def_role: str=None,
        col_properties_table: str=None, disc: str=None, bins: int=10):
        """
        Builds a Naive Bayes model.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        id_column : str
            the input table column identifying a unique instance id

        target_column : str
            the input table column representing the class

        in_columns : List[str], optional
            the input table columns with special properties, separated by a semi-colon (;).
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
            If the parameter is undefined, the input table column properties will be detected
            automatically.
            (Remark: colPropertiesTable with "COLROLE" column with value 'objweight' is unsupported,
            i.e. same as 'ignore')
            (Remark: colPropertiesTable with "COLWEIGHT" column with value '<wgt>' is unsupported,
            i.e. same as '1')

        disc : str, optional
            discretization type for numeric columns [ew, ef, em]

        bins : int, optional
            default number of bins for numeric columns
        """
        
        params = {
            'id': id_column,
            'target': target_column,
            'incolumn': in_columns,
            'coldeftype': col_def_type,
            'coldefrole': col_def_role,
            'colpropertiestable': col_properties_table,
            'disc': disc,
            'bins': bins
        }

        self._fit(in_df=in_df, params=params)
    
    def predict(self, in_df: IdaDataFrame, out_table: str=None, id_column: str=None,
        target_column: str=None, out_table_prob: str=None, mestimation: str=None):
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

        out_table_prob : str, optional
            if specified, the probability output table where class probability predictions
            will be stored

        mestimation : str, optional
            flag indicating to use m-estimation for probabilities. This kind of estimation of
            probabilities may be slower but can give better results for small or heavy
            unbalanced datasets.

        Returns
        -------
        IdaDataFrame
            the data frame containing row identifiers and predicted target values
        """

        params = {
            'id': id_column,
            'target': target_column,
            'outtableProb': out_table_prob,
            'mestimation': mestimation
            }

        return self._predict(in_df=in_df, params=params, out_table=out_table)
