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
from nzpyida.frame import IdaDataFrame
from nzpyida.analytics.utils import call_proc_df_in_out


def std_norm(in_df: IdaDataFrame, id_column: str, in_column: str, by_column: str=None,
    out_table: str=None) -> IdaDataFrame:
    """
    Standardization and normalization transformations use the original continuous
    attribute a to generate a new continuous attribute a ' that has a different range
    or distribution than the original attribute. Common transformations modify the
    range to fit the [-1,1 ] interval (normalization) or modify the distribution to
    have a mean of 0 and a standard deviation of 1 (standardization).

    This function normalize and stardardize columns of the input data frame and returns
    that in a new data frame.

    Parameters
    ----------
    in_df : IdaDataFrame
        the input data frame

    id_column : str
        the input table column identifying a unique instance id

    in_column : str
        the input table columns to consider, separated by a semi-colon (;).
        Each column name may be followed by :L to leave it unchanged, by :S to standardize
        its values, by :N to normalize its values or by :U to make it of unit length.
        Additionally, two columns may be indicated, separated by a slash (/), followed
        by :C to make the columns be a row unit vector or by :V to divide the column
        values by the length of the longest row vector.

    by_column : str, optional
        the input table column which splits the data into groups for which the operation
        is to be performed

    out_table : str, optional
        the output table with the modified data

    Returns
    -------
    IdaDataFrame
        the data frame with requested transformations
    """

    params = {
        'id': id_column,
        'incolumn': in_column,
        'by': by_column
    }
    return call_proc_df_in_out(proc='STD_NORM', in_df=in_df, params=params, out_table=out_table)


def impute_data(in_df: IdaDataFrame, in_column: str=None, method: str=None,
    numeric_value: float=-1, nominal_value: str='missing', out_table: str=None) -> IdaDataFrame:
    """
    Many analytic algorithms require that the data set has no missing attribute values.
    However, real-world data sets frequently suffer from missing attribute values.
    Missing value imputation provides usable attribute values in place of the missing values,
    allowing the algorithms to run.

    This function replaces missing values in the input data frame and returns the result
    in a new data frame.

    Parameters
    ----------
    in_df : IdaDataFrame
        the input data frame

    in_column : str, optional
        the input table column where missing values have to be replaced.
        If not specified, all input data columns are considered.

    method : str, optional
        the data imputation method. Allowed values are: mean, median, freq (most frequent value),
        replace. If not specified, the method is median for the numeric columns and freq for
        the nominal columns. The methods mean and median cannot be used with nominal columns.

    numeric_value : float, optional
        the numeric replacement value when method=replace

    nominal_value : str, optional
        the nominal replacement value when method=replace

    out_table : str, optional
        the output table with the modified data

    Returns
    -------
    IdaDataFrame
        the data frame with requested transformations
    """

    params = {
        'incolumn': in_column,
        'method': method,
        'numericvalue': numeric_value,
        'nominalvalue': nominal_value
    }
    return call_proc_df_in_out(proc='IMPUTE_DATA', in_df=in_df, params=params, out_table=out_table)


def random_sample(in_df: IdaDataFrame, size: int, fraction: float=None, by_column: str=None,
    out_signature: str=None, rand_seed: int=None, out_table: str=None) -> IdaDataFrame:
    """
    Random sampling procedures are a vital component of many analytical systems. They can
    be used to select a test sample and a training sample for a model building process
    (machine learning). They can also be used to get a smaller sample of the training
    set, which you may do because of learning algorithm complexity considerations.
    In both cases, you would sample without replacement.

    Another application of sampling is the learning methods based on bootstrapping.
    This requires many independent samples from the same data, which are preferentially
    applied if the available data sets are small or for other reasons where the sample
    independence is vital. Samples with replacement are usually drawn in this case.
    
    In application, sampling is used for promotion campaigns, for example when you want
    only a representative set of customers to be subjects of an action.
    In all cases, whether for use with scientific methods or business practices, uniform
    sampling is important.

    This function creates a random sample of a data frame a fixed size or a fixed
    probability and returns the result in a new data frame.

    Parameters
    ----------
    in_df : IdaDataFrame
        the input data frame

    size : int
        the number of rows in the sample (alias of size). If specified,
        the parameter <fraction> must not be specified. Only one of both parameters <num>
        and <size> must be specified.

    fraction : float
        the probability of each row to be in the sample. If specified, the parameters <num>
        and <size> must not be specified. Otherwise, one of both parameters <num> or <size>
        must be specified.

    by_column : str
        the column used to stratify the input table. If indicated, stratified sampling is
        done: it ensures that each value of the column is represented in the sample in
        about the same percentage as in the original input table.

    out_signature : str
        the input table columns to keep in the sample, separated by a semi-colon (;).
        If not specified, all columns are kept in the output table.

    rand_seed : int
        the seed of the random function

    out_table : str
        the output table with the modified data

    Returns
    -------
    IdaDataFrame
        the data frame with requested transformations
    """

    params = {
        'size': size,
        'fraction': fraction,
        'by': by_column,
        'outsignature': out_signature,
        'randseed': rand_seed
    }
    return call_proc_df_in_out(proc='RANDOM_SAMPLE', in_df=in_df, params=params, out_table=out_table)
