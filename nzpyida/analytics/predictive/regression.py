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
This module contains a class that is the base for all regression algorithms.
"""
from typing import Dict
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.utils import map_to_props, materialize_df, make_temp_table_name
from nzpyida.analytics.predictive.predictive_modeling import PredictiveModeling


class Regression(PredictiveModeling):
    """
    Base class for regression algorithms.
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
        self.score_proc = 'MSE'

    def predict(self, in_df: IdaDataFrame, out_table: str=None,
        id_column: str=None) -> IdaDataFrame:
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

        Returns
        -------
        IdaDataFrame
            the data frame containing row identifiers and predicted target values
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

    def score_all(self, in_df: IdaDataFrame, id_column: str,
        target_column: str) -> Dict[str, float]:
        """
        Scores the model using MSE, MAE, RSE and RAE. The model must exist.

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
        dict
            the model scores in a dictionary with MSE, MAE, RSE and RAE as keys
        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        out_table = make_temp_table_name()

        pred_view_needs_delete, true_view_needs_delete = False, False
        try:
            pred_df = self.predict(in_df=in_df, out_table=out_table, id_column=id_column)

            pred_view, pred_view_needs_delete = materialize_df(pred_df)
            true_view, true_view_needs_delete = materialize_df(in_df)

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

            res1 = pred_df.ida_query(f'call NZA..MSE(\'{params}\')')
            res2 = pred_df.ida_query(f'call NZA..MAE(\'{params}\')')
            res3 = pred_df.ida_query(f'call NZA..RSE(\'{params}\')')
            res4 = pred_df.ida_query(f'call NZA..RAE(\'{params}\')')
            res_dict = {
                "MSE": res1[0],
                "MAE": res2[0],
                "RSE": res3[0],
                "RAE": res4[0]
            }
            return res_dict
        finally:
            self.idadb.drop_table(out_table)
            if pred_view_needs_delete:
                self.idadb.drop_view(pred_view)
            if true_view_needs_delete:
                self.idadb.drop_view(true_view)
