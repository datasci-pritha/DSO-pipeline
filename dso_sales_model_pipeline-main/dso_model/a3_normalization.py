import pdb
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pickle import dump, load
from config_connection_varirables import logger
from sklearn.preprocessing import MinMaxScaler

input_file = 'dso_model/data/3_flooring_capping/train_sample_post_outlier_trtmnt.csv'

output_file = 'dso_model/data/7_post_transformation/train_sample.csv'
variable_file = 'dso_model/data/5_variable_shortlisting/variable_input_file.csv'

def normalize(df, sample):
    df_vars = pd.read_csv(variable_file)
    isFileValid = check_variable_file(df_vars)
    if isFileValid: 
        df_pred_vars = df_vars.loc[df_vars['TYPE'] == 'PREDICTOR']
        column_names_to_normalize = df_pred_vars['NAME'].tolist()

        x = df[column_names_to_normalize].values

        if sample == 'TRAIN':
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(x)
            dump(min_max_scaler, open('dso_model/data/4_normalization/train_sample_min_max_scaler.pkl', 'wb'))
            logger.debug("Saved the Training Sample's Normalization Object")
        elif sample == 'TEST':
            min_max_scaler = load(open('dso_model/data/4_normalization/train_sample_min_max_scaler.pkl', 'rb'))
        elif sample == 'BUILD':
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(x)
            dump(min_max_scaler, open('dso_model/data/4_normalization/build_sample_min_max_scaler.pkl', 'wb'))
            logger.debug("Saved the Build Sample's Normalization Object")
        elif sample == 'OUTSAMPLE':
            min_max_scaler = load(open('dso_model/data/4_normalization/build_sample_min_max_scaler.pkl', 'rb'))

        x_scaled = min_max_scaler.transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
        df[column_names_to_normalize] = df_temp
        return df
    else:
        print("Build or Update Variable File")
        sys.exit()

def build_variable_file(df):
    columns_index = ['NAME', 'TYPE']
    df_vars = pd.DataFrame(columns=columns_index)
    col_ind = 0
    none_type = ['invoice_yyyymm', 'next_2m_sales', 'next_3m_sales', 'next_6m_sales', 'balance_2m', 'balance_3m', 'avg_invoice_amount_2m',
                'group', 'age_of_obligor', 'total_no_of_invoices',  'snap_date', 'obligor_min_inv_date']
    for column in df.columns:
        if column in none_type:
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'NONE'
            col_ind = col_ind + 1
        elif column == 'obligor_code':
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'IDENTIFIER'
            col_ind = col_ind + 1
        elif column == 'dso_next_2m':
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'TARGET'
            col_ind = col_ind + 1
        else:
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'PREDICTOR'
            col_ind = col_ind + 1

    df_vars.to_csv(variable_file)
    print('Variable file written')

def check_variable_file(df_vars):
	gb = df_vars.groupby('TYPE')['NAME'].count()
	types_present = gb.index.tolist()
	is_valid_var_list = True
	
	if len(types_present) > 4:
		print("Unrecognized type - expect only 'IDENTIFIER', 'NONE', 'PREDICTOR', 'TARGET'")
		is_valid_var_list = False

	if 'IDENTIFIER' not in types_present:
		print("Invalid File: Identifier missing")
		is_valid_var_list = False				

	if 'PREDICTOR' not in types_present:
		print("Invalid File: Predictor missing")
		is_valid_var_list = False

	if 'TARGET' not in types_present:
		print("Invalid File: Target missing")
		is_valid_var_list = False

	if gb['TARGET']	> 1:
		print("Invalid File: More than one target")
		is_valid_var_list = False

	return is_valid_var_list

if __name__ == "__main__":
    # Load the Input File
    df = pd.read_csv(input_file)

    # Create the Variable File
    build_variable_file(df)
    logger.debug("Variable File Created")

    # Normalise the variables
    df_normalised = normalize(df, 'TRAIN')
    logger.debug("Normalized Train Sample")

    # Sort Values
    df_normalised.sort_values(['invoice_yyyymm', 'obligor_code'])

    # Save the Normalised File
    df_normalised.to_csv(output_file, index=False)