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
This module contains a class that is the base for all predictive algorithms.
"""
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.utils import map_to_props, materialize_df, make_temp_table_name
from nzpyida.analytics.utils import call_proc_df_in_out
from nzpyida.analytics.model_manager import ModelManager


class PredictiveModeling:
    """
    Generic class for predictive modeling algorithms.
    """

    def __init__(self, idadb: IdaDataBase, model_name: str):
        """
        Creates the predictive modeling class.

        Parameters
        ----------

        idada : IdaDataBase
            database connector

        model_name : str
            model name - if it exists in the database, it will be used, otherwise
            it must be trained using fit() function before prediction or scoring is called.
        """

        self.idadb = idadb
        self.model_name = model_name
        self.fit_proc = ''
        self.predict_proc = ''
        self.score_proc = ''
        self.score_inv = False
        self.target_column_in_output = None
        self.id_column_in_output = None

    def _fit(self, in_df: IdaDataFrame, params:dict):
        """
        Trains the model.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        params : dict
            the dictionary of attributes used to build the model
        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        ModelManager(self.idadb).drop_model(self.model_name)

        temp_view_name, need_delete = materialize_df(in_df)

        params['model'] = self.model_name
        params['intable'] = temp_view_name
        params_s = map_to_props(params)

        try:
            self.idadb.ida_query(f'call NZA..{self.fit_proc}(\'{params_s}\')')
        finally:
            if need_delete:
                self.idadb.drop_view(temp_view_name)

    def _predict(self, in_df: IdaDataFrame, params:dict, out_table: str=None) -> IdaDataFrame:
        """
        Makes predictions based on the model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        params : dict
            the dictionary of attributes used for making predictions

        out_table : str, optional
            the output table where the predictions will be stored

        Returns
        -------
        IdaDataFrame
            the data frame containing row identifiers and predicted target values
        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        params['model'] = self.model_name
        return call_proc_df_in_out(proc=self.predict_proc, in_df=in_df, params=params,
            out_table=out_table)

    def _score(self, in_df: IdaDataFrame, predict_params:dict, target_column: str) -> float:
        """
        Scores the model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        predict_params : dict
            the dictionary of attributes used for making predictions

        target_column : str
            the input table column representing the class

        Returns
        -------
        float
            the model score
        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        out_table = make_temp_table_name()

        pred_view_needs_delete, true_view_needs_delete = False, False
        try:

            pred_df = self._predict(in_df=in_df, params=predict_params, out_table=out_table)

            pred_view, pred_view_needs_delete = materialize_df(pred_df)
            true_view, true_view_needs_delete = materialize_df(in_df)

            id_column = predict_params.get('id')

            params = map_to_props({
                'pred_table': pred_view,
                'true_table': true_view,
                'pred_id': id_column if self.id_column_in_output is None
                    else self.id_column_in_output,
                'true_id': id_column,
                'pred_column': target_column if self.target_column_in_output is None
                    else self.target_column_in_output,
                'true_column': target_column
            })

            res = pred_df.ida_query(f'call NZA..{self.score_proc}(\'{params}\')')
            return 1-res[0] if self.score_inv else res[0]
        finally:
            self.idadb.drop_table(out_table)
            if pred_view_needs_delete:
                self.idadb.drop_view(pred_view)
            if true_view_needs_delete:
                self.idadb.drop_view(true_view)
