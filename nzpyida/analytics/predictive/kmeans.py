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
The k-means algorithm is the most widely-used clustering algorithm that uses
an explicit distance measure to partition the data set into clusters.
The main concept behind the k-means algorithm is to represent each cluster
by the vector of mean attribute values of all training instances assigned
to that cluster, called the cluster’s center. There are direct consequences
of such a cluster representation:

- the algorithm handles continuous attributes only, although workarounds
for discrete attributes are possible

- both the cluster formation and cluster modeling processes can be performed
in a computationally efficient way by applying the specified distance
function to match instances against cluster centers
"""

from typing import List
from nzpyida.frame import IdaDataFrame
from nzpyida.base import IdaDataBase
from nzpyida.analytics.utils import map_to_props, make_temp_table_name
from nzpyida.analytics.utils import get_auto_delete_context
from nzpyida.analytics.predictive.predictive_modeling import PredictiveModeling


class KMeans(PredictiveModeling):
    """
    KMeans clustering.
    """

    def __init__(self, idadb: IdaDataBase, model_name: str):
        """
        Creates the clusterer class.

        Parameters
        ----------

        idada : IdaDataBase
            database connector

        model_name : str
            model name - if it exists in the database, it will be used, otherwise
            it must be trained using fit() function before prediction or scoring is called.
        """

        super().__init__(idadb, model_name)
        self.fit_proc = 'KMEANS'
        self.predict_proc = 'PREDICT_KMEANS'
        self.score_proc = 'MSE'
        self.target_column_in_output = 'CLUSTER_ID'
        self.id_column_in_output = 'ID'

    def fit(self, in_df: IdaDataFrame, id_column: str, target_column: str,
        in_columns: List[str]=None, col_def_type: str=None, col_def_role: str=None,
        col_properties_table: str=None, out_table: str=None, distance: str='norm_euclidean',
        k: int=3, max_iter: int=5, rand_seed: int=12345, id_based: bool=False,
        statistics: str=None, transform: str='L') -> IdaDataFrame:
        """
        Creates and trains a model for clustering based on provided data and store
        it in a database.

        The training algorithm operates by performing several iterations of the same
        basic process. Each training instance is assigned to the closest cluster
        with respect to the specified distance function, applied to the instance
        and cluster center. All cluster centers are then re-calculated as the mean
        attribute value vectors of the instances assigned to particular clusters.
        The cluster centers are initialized by randomly picking k training instances,
        where k is the desired number of clusters. The iterative process should
        terminate when there are either no or sufficiently few changes in cluster
        assignments. In practice, however, it is sufficient to specify the number of
        iterations, typically a number between 3 and 36.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        id_column : str
            the input table column identifying a unique instance id

        target_column : str
            the input table column representing a class or a value to predict,
            this column is ignored by the Clustering algorithm.

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
            default role of the input table columns.
            Allowed values are 'input' and 'ignore'.
            If the parameter is undefined, all columns are considered 'input' columns.

        col_properties_table : str, optional
            the input table where column properties for the input table columns are stored.
            The format of this table is the output format of stored procedure
            nza..COLUMN_PROPERTIES().
            If the parameter is undefined, the input table column properties will be
            detected automatically.
            (Remark: colPropertiesTable with "COLROLE" column with value 'objweight'
            is unsupported, i.e. same as 'ignore')
            (Remark: colPropertiesTable with "COLWEIGHT" column with value '<wgt>' is unsupported,
            i.e. same as '1')

        out_table : str, optional
            the output table where clusters are assigned to each input table record

        distance : str, optional
            the distance function. Allowed values are: euclidean, norm_euclidean, manhattan,
            canberra, maximum, mahalanobis.

        k : int, optional
            number of centers

        max_iter : int, optional
            the maximum number of iterations to perform

        rand_seed : int, optional
            the random generator seed

        id_based : bool, optional
            the specification that random generator seed is based on id column value

        statistics : str, optional
            flags indicating which statistics to collect.
            Allowed values are: none, columns, values:n, all.
            If statistics=none, no statistics are collected.
            If statistics=columns, statistics on the input table columns like mean value are
            collec-ted.
            If statistics=values:n with n a positive number, statistics about the columns and
            the column values are collected. Up to <n> column value statistics are collected:
            If a nominal column contains more than <n> values, only the <n> most frequent
            column statistics are kept.
            If a numeric column contains more than <n> values, the values will be discretized and
            the statistics will be collected on the discretized values.
            Indicating statistics=all is equal to statistics=values:100.

        transform : str, optional
            flag indicating if the input table columns have to be transformed.
            Allowed values are: L (for leave as is), N (for normalization)
            or S (for standardization).
            If it is not specified, no transformation will be performed.

        """
        if not isinstance(in_df, IdaDataFrame):
            raise TypeError("Argument in_df should be an IdaDataFrame")

        auto_delete_context = None
        if not out_table:
            auto_delete_context = get_auto_delete_context('out_table')
            out_table = make_temp_table_name()

        params = {
            'id': id_column,
            'target': target_column,
            'incolumn': in_columns,
            'coldeftype': col_def_type,
            'coldefrole': col_def_role,
            'colpropertiestable': col_properties_table,
            'distance': distance,
            'k': k,
            'maxiter': max_iter,
            'randseed': rand_seed,
            'idbased': id_based,
            'statistics': statistics,
            'transform': transform,
            'outtable': out_table
        }

        self._fit(in_df=in_df, params=params)

        if auto_delete_context:
            auto_delete_context.add_table_to_delete(out_table)

        return IdaDataFrame(self.idadb, out_table)

    def predict(self, in_df: IdaDataFrame, out_table: str=None,
        id_column: str=None) -> IdaDataFrame:
        """
        Makes predictions based on this model. The model must exist.

        Parameters
        ----------
        in_df : IdaDataFrame
            the input data frame

        out_table : str, optional
            the output table where the assigned clusters will be stored

        id_column : str, optional
            the input table column identifying a unique instance id

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

    def __str__(self):
        params = map_to_props({'model': self.model_name})
        return self.idadb.ida_query(f'call NZA..PRINT_KMEANS(\'{params}\')')[0]
