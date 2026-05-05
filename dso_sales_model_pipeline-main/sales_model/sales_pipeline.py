import os
import pdb
import warnings
import numpy as np
import pandas as pd

from config_connection_varirables import logger, plot_graphs
from b1_filter_create_variables_data_splitting import preprocessing, exclusions_imputations_before_variable_creation
from b1_filter_create_variables_data_splitting import create_new_variables, create_build_outsample
from b1_filter_create_variables_data_splitting import exclusions_before_model_building, create_train_test_sample
from b2_flooring_capping import floor_cap, create_plots
from b3_normalization import build_variable_file, normalize
from b4_remove_multicollinearity import remove_multicollinearity
from b5_variable_selection import variable_selection
from b6_get_important_variables import get_imp_variables, build_final_variable_file
from b7_preprocess_test_data import preprocess_test_data
from b8_xgboost_model_built_on_train_validated_on_test import preprocess_data, sample_splitting, var_selection_xgb
from b8_xgboost_model_built_on_train_validated_on_test import floor_cap_target_n_save_files, create_all_plots, create_sales_matrix
from b9_final_xgboost_model_built_on_build_sample import preprocess_all_data, build_model

# Target Variable
target = 'next_2m_sales'

if __name__ == "__main__":
    # Load Input Data & Preprocess
    df, invoice_yyyymm, scored_obligors = preprocessing()

    # Exclusions and Imputations Before Variable Creation
    df = exclusions_imputations_before_variable_creation(df)

    # Variable Creation
    df = create_new_variables(df)

    # Create Build Sample & Outsample
    df_build, df_outsample = create_build_outsample(df)

    # Exclusions Before Model Building
    df_build = exclusions_before_model_building(df_build)

    # Create Train & Test Sample
    df_train, df_test = create_train_test_sample(df_build, scored_obligors)

    # Flooring and capping of Train Sample
    df_train_w_flooring_capping, col_list = floor_cap(df_train, 'TRAIN')

    if plot_graphs:
        # Plots the univarite graphs of the Variables
        create_plots(df_train, df_train_w_flooring_capping, col_list)
        logger.debug("Univariate Plots Created")

    # Build Variable & Target File
    build_variable_file(df)
    logger.debug("Variable File Created")

    # Normalize the Variables
    df_normalised = normalize(df_train_w_flooring_capping, 'TRAIN')
    logger.debug("Normalized Train Sample")

    # Remove Multicollinearity
    remove_multicollinearity(df_normalised)

    # Variable Selection
    variable_selection(df_normalised)

    # Get Important Variables
    predictor = get_imp_variables()

    # Build Final Variable File
    df_variable = build_final_variable_file(predictor, df_normalised)

    # Preprocess the Test Data
    df_test_preprocessed = preprocess_test_data(df_test)

    # Required Fields
    features_we_need = ['obligor_code', 'invoice_yyyymm', 'next_2m_sales']
    required_fields = predictor + features_we_need

    # Preprocess the Data - Remove rows with the Null
    df_train = df_normalised.copy()
    df_test = df_test_preprocessed.copy()
    df_null_treated_train_sample = preprocess_data(df_train[required_fields], 'TRAIN')
    df_null_treated_test_sample = preprocess_data(df_test[required_fields], 'TEST')

    # Sample Splitting
    X_train, y_train = sample_splitting(df_null_treated_train_sample, pd.Index(predictor), target)
    X_test, y_test = sample_splitting(df_null_treated_test_sample, pd.Index(predictor), target)

    # Variable Selection Using XGBoost & Model Building
    y_pred_train, y_pred_test, df_error = var_selection_xgb(X_train, X_test, y_train, y_test)

    # Floor and Cap the Target and Save the Files
    df_all, df_pred_train, df_pred_test = floor_cap_target_n_save_files(df_train, df_test, features_we_need, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test)

    if plot_graphs:
        # Create All Plots
        create_all_plots(df_all, df_pred_train)

    # Get the Confusion Matrix
    create_sales_matrix(df_pred_test)

    # Preprocess Build Sample & OOS
    df_null_treated, df_out_null_treated = preprocess_all_data(df_build, df_outsample)

    # Sample Splitting
    X, y  = sample_splitting(df_null_treated, pd.Index(predictor), target)
    X_out, y_out = sample_splitting(df_out_null_treated, pd.Index(predictor), target)

    # Build the Model & Save Files
    build_model(df_build, df_outsample, features_we_need, X, y, X_out, y_out)

    print('\nCode Complete')