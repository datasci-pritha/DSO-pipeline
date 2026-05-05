import pdb
import time
import warnings
import numpy as np
import pandas as pd
from pandas import ExcelWriter

from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# Input File
input_file = 'sales_model/data/7_post_transformation/train_sample.csv'


# Compute the vif for all given features
def compute_vif(df, considered_features):
    # Copy the DataFrame
    X = df[considered_features]

    # Add an Intercept
    # Calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # Create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Remove the Constant From the VIF DataFrame
    vif = vif[vif['Variable']!='intercept']

    return vif

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
        elif column == 'next_2m_sales':
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'TARGET'
            col_ind = col_ind + 1
        else:
            df_vars.loc[col_ind, 'NAME'] = column
            df_vars.loc[col_ind, 'TYPE'] = 'NONE'
            col_ind = col_ind + 1

    # Output File
    variable_output_file = 'sales_model/data/5_variable_shortlisting/variable_file.csv'
    df_vars.to_csv(variable_output_file)
    print('Variable file written')

    return

# Remove Multicollinearity
def remove_multicollinearity(df):
    # Output File
    variable_vif_file = 'sales_model/data/5_variable_shortlisting/vif.xlsx'
    variable_vif_ = 'sales_model/data/5_variable_shortlisting/variable_vif.csv'

    # Target
    target = 'next_2m_sales'

    # Get All the Features
    considered_features = df.columns.tolist()[2:]
    considered_features = considered_features[:considered_features.index(target)] + considered_features[considered_features.index(target)+1:]
    print('\nNumber of Variables: ', len(considered_features))

    # Get the VIF
    vif = compute_vif(df, considered_features)

    # Remove all Variables with VIF greater than 20
    short_listed_variables = vif['Variable'][vif['VIF'] < 20].to_list()
    print('\nNumber of Variables with VIF less than 20: ', len(short_listed_variables))

    # Get the VIF
    vif_w_lt_20 = compute_vif(df, short_listed_variables)

    # Remove all Variables with VIF greater than 15
    short_listed_variables = vif_w_lt_20['Variable'][vif_w_lt_20['VIF'] < 15].to_list()
    print('\nNumber of Variables with VIF less than 15 after recalculating VIF for variables with VIF less than 20: ', len(short_listed_variables))

    # Get the VIF
    vif_w_lt_15 = compute_vif(df, short_listed_variables)

    # Remove all Variables with VIF greater than 10
    short_listed_variables = vif_w_lt_15['Variable'][vif_w_lt_15['VIF'] <= 10].to_list()
    print('\nNumber of Variables with VIF less than 10 after recalculating VIF for variables with VIF less than 15: ', len(short_listed_variables))

    # Create a Dataframe to get the sum of all VIF's while removing a variable
    df_short_list = pd.DataFrame(columns=['variable_removed', 'sum_of_vif', 'max_vif'])

    # Removal of Variables with VIF greater than 5 
    variables_to_remove = vif_w_lt_15['Variable'][(vif_w_lt_15['VIF'] > 5) & (vif_w_lt_15['VIF'] <= 10)].to_list()
    print('\nNumber of Variables that are in consideration for Removal: ', len(variables_to_remove))

    # Flag
    min_vif = 0
    least_vif_df = pd.DataFrame()

    # Remove each variable & calculate VIF
    for var in variables_to_remove:
        print('\nRemoving Variable: ', var)
        list_of_features = short_listed_variables.copy()
        list_of_features.remove(var)

        # Compute VIF with the rest of the Features
        temp_vif = compute_vif(df, list_of_features)
        sum_of_vif = temp_vif.VIF.sum()
        data = {'variable_removed':var, 'sum_of_vif':sum_of_vif, 'max_vif':temp_vif.VIF.max()}

        # Append to the Metrics DataFrame
        # df_short_list = df_short_list.append(data, ignore_index=True)
        df_short_list = pd.concat([df_short_list, pd.DataFrame([data])], ignore_index=True)

        # Getting the VIF for Minimum Sum of VIF
        if min_vif == 0:
            min_vif = sum_of_vif
            least_vif_df = temp_vif
        elif min_vif > sum_of_vif:
            min_vif = sum_of_vif
            least_vif_df = temp_vif
        
        if len(variables_to_remove) > 3:
            new_variables_to_remove = variables_to_remove.copy()
            new_variables_to_remove.remove(var)
            for second_var in new_variables_to_remove:
                print('\nRemoving Variable: ', second_var)
                temp = list_of_features.copy()
                temp.remove(second_var)
                new_variables_to_remove.remove(second_var)
                temp_vif = compute_vif(df, temp)
                data = {'variable_removed':[var, second_var], 'sum_of_vif':temp_vif.VIF.sum(), 'max_vif':temp_vif.VIF.max()}
                # Append to the Metrics DataFrame
                # df_short_list = df_short_list.append(data, ignore_index=True)
                df_short_list = pd.concat([df_short_list, pd.DataFrame([data])], ignore_index=True)

                # Getting the VIF for Minimum Sum of VIF
                if min_vif == 0:
                    min_vif = sum_of_vif
                    least_vif_df = temp_vif
                elif min_vif > sum_of_vif:
                    min_vif = sum_of_vif
                    least_vif_df = temp_vif

                for third_var in new_variables_to_remove:
                    print('\nRemoving Variable: ', third_var)
                    temp_ = temp.copy()
                    temp_.remove(third_var)
                    temp_vif = compute_vif(df, temp_)
                    data = {'variable_removed':[var, second_var, third_var], 'sum_of_vif':temp_vif.VIF.sum(), 'max_vif':temp_vif.VIF.max()}
                    # Append to the Metrics DataFrame
                    # df_short_list = df_short_list.append(data, ignore_index=True)
                    df_short_list = pd.concat([df_short_list, pd.DataFrame([data])], ignore_index=True)

                    # Getting the VIF for Minimum Sum of VIF
                    if min_vif == 0:
                        min_vif = sum_of_vif
                        least_vif_df = temp_vif
                    elif min_vif > sum_of_vif:
                        min_vif = sum_of_vif
                        least_vif_df = temp_vif

    # Get the variable which reduced the sum_of_vif the most & Remove it
    variable_to_be_removed = df_short_list['variable_removed'][df_short_list['sum_of_vif'] == df_short_list['sum_of_vif'].max()].values[0]
    short_listed_variables.remove(variable_to_be_removed)
    print('\nFinally Removed Variable: ', variable_to_be_removed)
    print('\nFeatures Considered: ', variable_to_be_removed)

    # Rebuild the Variable File
    build_variable_file(short_listed_variables)

    # Save the VIF Values
    with ExcelWriter(variable_vif_file) as writer:
        vif.to_excel(writer,'VIF - All Variables')
        vif_w_lt_20.to_excel(writer,'VIF - Less than 20')
        vif_w_lt_15.to_excel(writer,'VIF - Less than 15')

    # VIF
    df_short_list.to_csv(variable_vif_, index=False)

    print('\n Removed Multicollinearity')

    return

if __name__ == "__main__":
    # Load the Input Files
    df = pd.read_csv(input_file)

    # Remove Multicollinearity
    remove_multicollinearity(df)