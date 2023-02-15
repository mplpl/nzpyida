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
This module contains a class that is the base for all classification algorithms.
"""
from typing import Tuple
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.utils import map_to_props, materialize_df, make_temp_table_name
from nzpyida.analytics.utils import get_auto_delete_context
from nzpyida.analytics.predictive.predictive_modeling import PredictiveModeling


class Classification(PredictiveModeling):
    """
    Base class for classification algorithms.
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
        self.target_column_in_output = 'CLASS'
        self.id_column_in_output = 'ID'
        self.score_proc = 'CERROR'
        self.score_inv = True

    def predict(self, in_df: IdaDataFrame, out_table: str=None,
        id_column: str=None) -> IdaDataFrame:
        """
        Makes predictions based on this model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame for predictions

        out_table : str, optional
            the output table where the predictions will be stored

        id_column : str, optional
            the input table column identifying a unique instance id
        """

        params = {
            'id': id_column
        }

        return self._predict(in_df=in_df, params=params, out_table=out_table)

    def score(self, in_df: IdaDataFrame, id_column: str, target_column: str) -> float:
        """
        Scores the model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame for scoring

        id_column : str
            the input table column identifying a unique instance id

        target_column : str
            the input table column representing the class

        Returns
        -------
        float
            the model score
        """

        params = {
            'id': id_column
        }

        return self._score(in_df=in_df, predict_params=params, target_column=target_column)

    def conf_matrix(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        out_matrix_table: str=None) -> Tuple[IdaDataFrame, float, float]:
        """
        Makes a predition for a test data set given by the user and returns a confusion matrix,
        together with other stats (ACC and WACC).

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame for scoring

        id_column : str
            the input table column identifying a unique instance id

        target_column : str
            the input table column representing the class

        out_matrix_table : str, optional
            the output table where the confidence matrix will be stored

        Returns
        -------
        IdaDataFrame
            the confidence matrix data frame

        float
            classification accuracy (ACC)

        float
            weighted classification accuracy (WACC)
        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        out_table = make_temp_table_name()

        pred_view_needs_delete, true_view_needs_delete = False, False
        try:
            pred_df = self.predict(in_df=in_df, out_table=out_table, id_column=id_column)

            pred_view, pred_view_needs_delete = materialize_df(pred_df)
            true_view, true_view_needs_delete = materialize_df(in_df)

            auto_delete_context = None
            if not out_matrix_table:
                auto_delete_context = get_auto_delete_context('out_matrix_table')
                out_matrix_table = make_temp_table_name()

            params = map_to_props({
                'resulttable': pred_view,
                'intable': true_view,
                'resultid': 'ID',
                'id': id_column,
                'resulttarget': 'CLASS',
                'target': target_column,
                'matrixTable': out_matrix_table
            })
            pred_df.ida_query(f'call NZA..CONFUSION_MATRIX(\'{params}\')')

            if auto_delete_context:
                auto_delete_context.add_table_to_delete(out_matrix_table)

            out_df = IdaDataFrame(self.idadb, out_matrix_table)

            params = map_to_props({
                'matrixTable': out_matrix_table
            })

            res_acc = pred_df.ida_query(f'call NZA..CMATRIX_ACC(\'{params}\')')
            res_wacc = pred_df.ida_query(f'call NZA..CMATRIX_WACC(\'{params}\')')

            return out_df, res_acc[0], res_wacc[0]

        finally:
            self.idadb.drop_table(out_table)
            if pred_view_needs_delete:
                self.idadb.drop_view(pred_view)
            if true_view_needs_delete:
                self.idadb.drop_view(true_view)
