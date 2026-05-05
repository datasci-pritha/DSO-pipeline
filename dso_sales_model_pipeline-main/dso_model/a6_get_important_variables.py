import pdb
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from config_connection_varirables import logger, save_intermediate_files

# Input Files
input_file = 'dso_model/data/7_post_transformation/train_sample.csv'
variables_file = 'dso_model/data/6_random_forest/var_importance_.csv'

# Output Files
variable_output_file = 'dso_model/data/7_post_transformation/variable_file.csv'

# Rebuild Variable File
def build_variable_file(variable):
    df = pd.read_csv(input_file)
    columns_index = ['NAME', 'TYPE']
    df_vars = pd.DataFrame(columns=columns_index)
    col_ind = 0
    for column in df.columns:
        if column in variable:
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'PREDICTOR'
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
            df_vars.loc[col_ind, 'TYPE'] = 'NONE'
            col_ind = col_ind + 1

    df_vars.to_csv(variable_output_file)
    print('Variable file written')

    return

if __name__ == "__main__":
    df = pd.read_csv(variables_file)

    variables = df['variable'].to_list()
    # variables = variables[0:variables.index('RANDOM')] if variables.index('RANDOM') < 6 else variables[0:6]
    variables = variables[0:variables.index('RANDOM')]

    predictor = list(set(variables))
    logger.debug("Variables That Came out Significant \n %s", predictor)

    build_variable_file(predictor)
    logger.debug("Variable File Created For Modeling")

    print("Code Completed")