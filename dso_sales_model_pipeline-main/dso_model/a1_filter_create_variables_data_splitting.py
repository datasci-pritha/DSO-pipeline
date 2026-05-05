import os
import pdb
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from config_connection_varirables import logger

from model_inputs_dissection import obligors_to_score

warnings.filterwarnings('ignore')

build_sample = 'dso_model/data/2_combined/build_sample.csv'
train_sample = 'dso_model/data/2_combined/train_sample.csv'
test_sample = 'dso_model/data/2_combined/test_sample.csv'
outsample_file = 'dso_model/data/2_combined/outsample.csv'
whole_data = 'dso_model/data/2_combined/whole_data.csv'

# Load Input Data & Preprocess
def preprocessing():
    # Load Input Data
    predictors_file = os.getcwd() + '/Data/Model Inputs/payment_behaviour_table.csv'
    raw_file = os.getcwd() + '/Data/Model Inputs/sales_forecast_data.csv'

    df_raw = pd.read_csv(raw_file)
    predictors = pd.read_csv(predictors_file)
    if df_raw.columns[0] == 'Unnamed: 0':
        df_raw.drop(df_raw.columns[0], axis=1, inplace = True)
    if predictors.columns[0] == 'Unnamed: 0':
        predictors.drop(predictors.columns[0], axis=1, inplace = True)

    # Drop the Fields from Payment Behaviour File
    drop_vars_predictors = ['num_invoices_3m','num_invoices_6m', 'days_outs_3m_avg', 'days_outs_6m_avg']
    predictors.drop(drop_vars_predictors, axis = 1,inplace = True)

    # Drop the Fields from Sales Forecast File
    drop_vars_raw = ['balance_3m', 'next_3m_sales', 'next_6m_sales']
    df_raw.drop(drop_vars_raw, axis = 1,inplace = True)

    # Obligors to Be Scored
    scored_obligors = obligors_to_score(df_raw[['obligor_code', 'invoice_yyyymm', 'sales', 'collections', 'balance', 'beyond_max_snap']])

    # Merging paid variable file and sales variable file
    df_merged = pd.merge(df_raw, predictors, how='left', left_on=['obligor_code','invoice_yyyymm'], right_on = ['obligor_code', 'invoice_yyyymm'])
    df_merged['invoice_yyyymm'] = df_merged['invoice_yyyymm'].astype(object)
    df_merged = df_merged.sort_values(['obligor_code','invoice_yyyymm'])

    invoice_yyyymm = pd.Series(df_merged['invoice_yyyymm'].unique()).sort_values().to_list()

    return df_merged, invoice_yyyymm, scored_obligors

# Exclusions and Imputations Before Variable Creation
def exclusions_imputations_before_variable_creation(df_merged):
    print("\n\nBefore All Exclusion: ", df_merged.shape)
    print("Obligor Count Before All Exclusion: ", df_merged.obligor_code.nunique())
    logger.debug("\n\nBefore All Exclusion: %s", df_merged.shape)
    logger.debug("Obligor Count Before All Exclusion: %s", df_merged.obligor_code.nunique())

    # Removing Rows with Negative and Zero Sales Till Now (i.e sales_till_now >= 0)
    df_merged = df_merged[df_merged['sales_till_now'] >= 0]
    print("\nShape after Removing Negative sales_till_now: ", df_merged.shape)
    print("Obligor Count Shape after Removing Negative sales_till_now: ", df_merged.obligor_code.nunique())
    logger.debug("\nShape after Removing Negative sales_till_now: %s", df_merged.shape)
    logger.debug("Obligor Count Removing Negative sales_till_now: %s", df_merged.obligor_code.nunique())
    
    # Remove Rows with Negative Balance (i.e balance < 0)
    print("\nNumber of rows with Negative balance: ", df_merged[df_merged['balance'] < 0].shape)
    print("Obligor Count with Negative balance: ", df_merged[df_merged['balance'] < 0].obligor_code.nunique())
    logger.debug("\nNumber of rows with Negative balance: %s", df_merged[df_merged['balance'] < 0].shape)
    logger.debug("Obligor Count with Negative balance: %s", df_merged[df_merged['balance'] < 0].obligor_code.nunique())
    df_merged['balance'] = np.where(df_merged['balance'] < 0, 0, df_merged['balance'])

    # Remove Rows with Negative Next 2 Months Balanace (i.e balance_2m < 0)
    print("\nNumber of rows with Negative balance_2m: ", df_merged[df_merged['balance_2m'] < 0].shape)
    print("Obligor Count with Negative balance_2m: ", df_merged[df_merged['balance_2m'] < 0].obligor_code.nunique())
    logger.debug("\nNumber of rows with Negative balance_2m: %s", df_merged[df_merged['balance'] < 0].shape)
    logger.debug("Obligor Count with Negative balance_2m: %s", df_merged[df_merged['balance'] < 0].obligor_code.nunique())
    df_merged['balance_2m'] = np.where(df_merged['balance_2m'] < 0, 0, df_merged['balance_2m'])
    print("\nShape after Removing Negative balance & balance_2m: ", df_merged.shape)
    print("Obligor Count Shape after Removing Negative balance & balance_2m: ", df_merged.obligor_code.nunique())
    logger.debug("\nShape after Removing Negative balance & balance_2m: %s", df_merged.shape)
    logger.debug("Obligor Count Removing Negative balance & balance_2m: %s", df_merged.obligor_code.nunique())

    df_merged = df_merged.reset_index(drop=True)
    logger.debug("\n\nExclusions and Imputations before Variable Creation Done")
    return df_merged

def get_remaining_receivables(df, current_receivable, counter, snap_count):
    # Get the Remaining Receivables
    remaining_receivables = current_receivable
    if counter <= snap_count:
        # Get the Sales for the Snap
        sales = df['sales'].iat[snap_count - counter - 1]

        # Update the Remaining Receivables
        remaining_receivables = 0 if remaining_receivables - sales <= 0 else remaining_receivables - sales
    else:
        sales = np.nan
        remaining_receivables = np.nan

    return remaining_receivables, sales, current_receivable
    
def get_exhaustive_dso(df):
    # Change Dtypes
    df['invoice_yyyymm'] = df['invoice_yyyymm'].astype(int)

    # Get the Data With Receivables > 0 & Sales Till Now > 0
    df_ = df[(df['balance']>0)&(df['sales_till_now']>0)].reset_index(drop=True)
    
    # Get the obligor list with Receivables > 0 & Sales Till Now > 0
    obligors = df_.obligor_code.unique().tolist()

    # Dataframe to store DSO for Receivables > 0
    ex_dso = pd.DataFrame(columns=['obligor_code', 'invoice_yyyymm', 'exhaustive_dso'])

    for obl in tqdm(obligors):
        # print('Customer Considered: ', obl)
        # Get the Snaps in Order
        snaps = df_[df_['obligor_code']==obl].invoice_yyyymm.unique().tolist()
        snaps.sort()
        # snaps = list(map(str, snaps))
        for snap in snaps:
            # Flag
            counter = 0

            # Exhaustive DSO Variable
            dso = 0

            # Get the Data of that customer till the current snap
            df_obl = df[['obligor_code', 'invoice_yyyymm', 'balance', 'sales']][(df['obligor_code']==obl)&(df['invoice_yyyymm']<=snap)].reset_index(drop=True).sort_values(['obligor_code', 'invoice_yyyymm'])
            snap_count = df_obl.shape[0]

            # Current Receivables
            # current_receivable = df_obl[df_obl['invoice_yyyymm']==snap].balance.values[0]
            current_receivable = df_obl['balance'].to_numpy()[df_obl['invoice_yyyymm'].to_numpy() == snap].item()

            # Track the Receivables
            remaining_receivables = current_receivable

            while (remaining_receivables > 0 and counter < 13):
                # Get the Remaining Receivables
                remaining_receivables, sales, balance = get_remaining_receivables(df_obl, remaining_receivables, counter, snap_count)

                # Based on the remaining receivables, calculate the DSO
                if (remaining_receivables > 0 and counter < 13):
                    dso = dso + 30
                else:
                    if counter >= 12:
                        dso = 360
                    elif remaining_receivables == np.nan:
                        dso = np.nan
                    else:
                        dso = dso + (balance * 30) / sales
                    # Append Data to the Exhaustive DSO DataFrame
                    temp = {'obligor_code':obl, 'invoice_yyyymm':snap, 'exhaustive_dso':dso}
                    ex_dso = ex_dso._append(temp, ignore_index=True)
                    break
                
                # Update the counter
                counter = counter + 1

    # Join the Exhaustive DSO Dataframe with the Original DataFrame
    df = pd.merge(df, ex_dso, how='left', left_on=['obligor_code','invoice_yyyymm'], right_on = ['obligor_code', 'invoice_yyyymm'])
    
    # Combine the Exhausitve DSO
    df['exhaustive_dso'] = np.where(df['balance'] <= 0, 0, df['exhaustive_dso'])

    # Change Dtypes
    df['invoice_yyyymm'] = df['invoice_yyyymm'].astype(object)
    df['exhaustive_dso'] = df['exhaustive_dso'].astype(float)

    # Sort the Dataframe
    df = df.sort_values(['obligor_code', 'invoice_yyyymm'])

    return df

# Variable Creation
def create_new_variables(df_merged):
    # Segments in Obligors
    df_merged['obligor_code'] = df_merged['obligor_code'].astype(str)
    df_merged['group'] = df_merged['obligor_code'].str.slice(0,2)
    df_merged['group'] = df_merged['group'].astype(int)
    df_merged['trade_flag'] = np.where(df_merged['group'].isin([50,60]),1,0)
    df_merged['govt_flag'] = np.where(df_merged['group'].isin([40]),1,0)
    df_merged['ib_flag'] = np.where(df_merged['group'].isin([70]),1,0)
    df_merged['obligor_code'] = df_merged['obligor_code'].astype(int)

    df_merged['invoice_yyyymm'] = df_merged['invoice_yyyymm'].astype(object)
    df_merged['cna_flag'] = np.where(df_merged['group'].isin(['50']), 1, 0)    

    df_merged = df_merged.sort_values(['obligor_code','invoice_yyyymm'])

    # Month Level Variables
    df_merged['month'] = df_merged['invoice_yyyymm']%100
    df_merged['quarter_ending'] = np.where(df_merged['month'].isin([3, 6, 9, 12]), 1, 0)
    df_merged['month'] = 'month_' + df_merged['month'].astype(str)
    dummies = pd.get_dummies(df_merged[['month']])
    df_merged = pd.concat([df_merged, dummies], axis=1)

    # Invoice Variables
    df_merged['invoices_per_month'] = df_merged['no_of_invoices_till_now'] * 30 / df_merged['obligor_age'] 
    df_merged['invoices_per_month'] = np.where(df_merged['invoices_per_month'].isin([np.inf, np.nan, -np.inf]), df_merged['no_of_invoices_till_now'], df_merged['invoices_per_month'])
    df_merged['avg_invoice_amount'] = np.where(df_merged['no_of_invoices_till_now'] > 0, df_merged['sales_till_now'] / df_merged['no_of_invoices_till_now'], 0)
    df_merged['avg_invoice_amount_per_month'] = np.where(df_merged['obligor_age'] > 0, df_merged['sales_till_now'] * 30 / df_merged['obligor_age'], df_merged['sales_till_now'])
    df_merged['avg_invoice_amount_per_month'] = np.where(df_merged['avg_invoice_amount_per_month'].isin([np.inf]), df_merged['sales_till_now'], df_merged['avg_invoice_amount_per_month'])

    # Future Invoice Amount Variables
    df_merged['avg_invoice_amount_2m'] = df_merged.groupby(['obligor_code'])['avg_invoice_amount_per_month'].transform(lambda x: x.shift(-2)).round(decimals=0)
    df_merged['avg_invoice_amount_m_imputed'] = df_merged.groupby(['obligor_code'])['avg_invoice_amount_per_month'].transform(lambda x: x.shift(-1)).round(decimals=0)
    df_merged['avg_invoice_amount_m_imputed'] = np.where(df_merged['avg_invoice_amount_m_imputed'].isna(), df_merged['avg_invoice_amount_per_month'], df_merged['avg_invoice_amount_m_imputed'])
    df_merged['avg_invoice_amount_2m_imputed'] = df_merged.groupby(['obligor_code'])['avg_invoice_amount_per_month'].transform(lambda x: x.shift(-2)).round(decimals=0)
    df_merged['avg_invoice_amount_2m_imputed'] = np.where(df_merged['avg_invoice_amount_2m_imputed'].isna(), np.where(df_merged['avg_invoice_amount_m_imputed'].isna(), 0, df_merged['avg_invoice_amount_m_imputed']), df_merged['avg_invoice_amount_2m_imputed'])
    df_merged['avg_invoice_amount_2m']  = np.where(df_merged['avg_invoice_amount_2m'].isna(),df_merged['avg_invoice_amount_2m_imputed'],df_merged['avg_invoice_amount_2m'])  
    
    # Sales Variables
    df_merged['sales_flag'] = np.where(df_merged['sales'] > 0, 1, 0)
    df_merged['no_of_months_sales'] = df_merged.groupby(['obligor_code'])['sales_flag'].cumsum()
    df_merged['sales_12m'] = df_merged.groupby(['obligor_code'])['sales'].transform(lambda x: x.rolling(12, min_periods=1).sum()).round(decimals=0)
    df_merged['sales_24m'] = df_merged.groupby(['obligor_code'])['sales'].transform(lambda x: x.rolling(24, min_periods=1).sum()).round(decimals=0)
    df_merged['sales_2m_1'] = df_merged['sales_1'] + df_merged['sales_2']     # sales_2m_1 = sales in the previous month + sales in 2nd previous month
    df_merged['sales_2m_2'] = df_merged['sales_2'] + df_merged['sales_3']     # sales_2m_2 = sales in 2nd previous month + sales in 3rd previous month
    df_merged['SALES_4M'] = df_merged['sales'] + df_merged['sales_1'] + df_merged['sales_2'] + df_merged['sales_3'] 
    df_merged['AVG_SALES'] = df_merged.groupby(['obligor_code'])['sales'].transform(lambda x: x.expanding().mean()).round(decimals = 0)
    df_merged['AVG_SALES_LAST_6M'] = df_merged.groupby(['obligor_code'])['sales'].transform(lambda x: x.rolling(6).mean()).round(decimals=0)

    df_merged['sales_2m_1'] = np.where(df_merged['sales_2m_1'] <= 0, 0, df_merged['sales_2m_1'])
    df_merged['sales_2m_2'] = np.where(df_merged['sales_2m_2'] <= 0, 0, df_merged['sales_2m_2'])
    df_merged['AVG_SALES_LAST_6M'] = np.where(df_merged['AVG_SALES_LAST_6M'].isna(),df_merged['AVG_SALES'],df_merged['AVG_SALES_LAST_6M'])

    # Sales 2M Variables
    df_merged['AVG_SALES_2M_LAST_6M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.rolling(6).mean())
    df_merged['AVG_SALES_2M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.expanding().mean())
    df_merged['MAX_SALES_2M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.expanding().max())
    df_merged['MIN_SALES_2M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.expanding().min())
    df_merged['90TH_SALES_2M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.expanding().quantile(0.9))
    df_merged['10TH_SALES_2M'] = df_merged.groupby(['obligor_code'])['sales_2m'].transform(lambda x: x.expanding().quantile(0.1))
    df_merged['AVG_SALES_2M_LAST_6M'] = np.where(df_merged['AVG_SALES_2M_LAST_6M'].isna(),df_merged['AVG_SALES_2M'],df_merged['AVG_SALES_2M_LAST_6M'])

    # Sales Growth Variables
    df_merged['sales_growth'] = df_merged['sales_2m']/(df_merged['sales_2m_2'] + 1)
    df_merged['AVG_SALES_GROWTH'] = df_merged.groupby(['obligor_code'])['sales_growth'].transform(lambda x: x.expanding().mean())
    df_merged['MAX_SALES_GROWTH'] = df_merged.groupby(['obligor_code'])['sales_growth'].transform(lambda x: x.expanding().max())
    df_merged['MIN_SALES_GROWTH'] = df_merged.groupby(['obligor_code'])['sales_growth'].transform(lambda x: x.expanding().min())
    df_merged['90TH_SALES_GROWTH'] = df_merged.groupby(['obligor_code'])['sales_growth'].transform(lambda x: x.expanding().quantile(0.9))
    df_merged['10TH_SALES_GROWTH'] = df_merged.groupby(['obligor_code'])['sales_growth'].transform(lambda x: x.expanding().quantile(0.1))

    # Collection Variables
    df_merged['collections_flag'] = np.where(df_merged['collections'] > 0, 1, 0)
    df_merged['no_of_months_collections'] = df_merged.groupby(['obligor_code'])['collections_flag'].cumsum()
    df_merged['collection_last_m'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.shift(1)).round(decimals=0)
    df_merged['AVG_COLLECTION_LAST_6M'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.rolling(6).mean())
    df_merged['AVG_COLLECTION'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.expanding().mean())
    df_merged['MAX_COLLECTION'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.expanding().max())
    df_merged['MIN_COLLECTION'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.expanding().min())
    df_merged['90TH_COLLECTION'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.expanding().quantile(0.9))
    df_merged['10TH_COLLECTION'] = df_merged.groupby(['obligor_code'])['collections'].transform(lambda x: x.expanding().quantile(0.1))
    df_merged['AVG_COLLECTION_LAST_6M'] = np.where(df_merged['AVG_COLLECTION_LAST_6M'].isna(),df_merged['AVG_COLLECTION'],df_merged['AVG_COLLECTION_LAST_6M'])

    # Collection Growth Variables
    df_merged['collection_growth'] = df_merged['collections']/(df_merged['collection_last_m'] + 1)
    df_merged['AVG_COLLECTION_GROWTH'] = df_merged.groupby(['obligor_code'])['collection_growth'].transform(lambda x: x.expanding().mean())
    df_merged['MAX_COLLECTION_GROWTH'] = df_merged.groupby(['obligor_code'])['collection_growth'].transform(lambda x: x.expanding().max())
    df_merged['MIN_COLLECTION_GROWTH'] = df_merged.groupby(['obligor_code'])['collection_growth'].transform(lambda x: x.expanding().min())
    df_merged['90TH_COLLECTION_GROWTH'] = df_merged.groupby(['obligor_code'])['collection_growth'].transform(lambda x: x.expanding().quantile(0.9))
    df_merged['10TH_COLLECTION_GROWTH'] = df_merged.groupby(['obligor_code'])['collection_growth'].transform(lambda x: x.expanding().quantile(0.1))

    # Balance Variables
    df_merged['AVG_Balance_LAST_6M'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.rolling(6).mean()).round(decimals=0)
    df_merged['AVG_Balance'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().mean()).round(decimals=0)
    df_merged['MAX_Balance'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().max()).round(decimals=0)
    df_merged['MIN_Balance'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().min()).round(decimals=0)
    df_merged['90TH_Balance'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().quantile(0.9)).round(decimals=0)
    df_merged['10TH_Balance'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().quantile(0.1)).round(decimals=0)
    # df_merged['Balance_STD'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.expanding().std()).round(decimals=0)
    df_merged['AVG_Balance_LAST_6M'] = np.where(df_merged['AVG_Balance_LAST_6M'].isna(),df_merged['AVG_Balance'],df_merged['AVG_Balance_LAST_6M'])

    # Future Balance Variables
    df_merged['balance_m_imputed'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.shift(-1)).round(decimals=0)
    df_merged['balance_m_imputed'] = np.where(df_merged['balance_m_imputed'].isna(), 0, df_merged['balance_m_imputed'])
    df_merged['balance_2m_imputed'] = df_merged.groupby(['obligor_code'])['balance'].transform(lambda x: x.shift(-2)).round(decimals=0)
    df_merged['balance_2m_imputed'] = np.where(df_merged['balance_2m_imputed'].isna(), np.where(df_merged['balance_m_imputed'].isna(), 0, df_merged['balance_m_imputed']), df_merged['balance_2m_imputed'])
    df_merged['balance_2m']  = np.where(df_merged['balance_2m'].isna(),df_merged['balance_2m_imputed'],df_merged['balance_2m'])   

    # DSO Variables
    df_merged['days_outs_24m_avg'] = np.where(df_merged['balance'] <= 0, 0, np.where(df_merged['sales_24m'] > 0, (df_merged['balance']/df_merged['sales_24m'])*720, np.nan))
    df_merged['days_outs_12m_avg'] = np.where(df_merged['balance'] <= 0, 0, np.where(df_merged['sales_12m'] > 0, (df_merged['balance']/df_merged['sales_12m'])*360, df_merged['days_outs_24m_avg']))
    df_merged['days_outs_6m_avg'] = np.where(df_merged['balance'] <= 0, 0, np.where(df_merged['sales_6m'] > 0, (df_merged['balance']/df_merged['sales_6m'])*180, df_merged['days_outs_12m_avg']))
    df_merged['days_outs_3m_avg'] = np.where(df_merged['balance'] <= 0, 0, np.where(df_merged['sales_3m'] > 0, (df_merged['balance']/df_merged['sales_3m'])*90, df_merged['days_outs_6m_avg']))
    df_merged['days_outs_2m_avg'] = np.where(df_merged['balance'] <= 0, 0, np.where(df_merged['sales_2m'] > 0, (df_merged['balance']/df_merged['sales_2m'])*60, df_merged['days_outs_6m_avg']))
    df_merged['dso_prev2m_1'] = np.where(df_merged['balance_1'] <= 0, 0, np.where(df_merged['sales_2m_1'] > 0, (df_merged['balance_1']/df_merged['sales_2m_1'])*60,df_merged['days_outs_2m_avg']))

    df_merged['AVG_DSO_LAST_6M'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.rolling(6, min_periods=1).mean()).round(decimals=0)
    df_merged['AVG_DSO'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding(min_periods=1).mean()).round(decimals=0)
    df_merged['MAX_DSO'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding(min_periods=1).max()).round(decimals=0)
    df_merged['MIN_DSO'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding(min_periods=1).min()).round(decimals=0)
    df_merged['90TH_DSO'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding(min_periods=1).quantile(0.9)).round(decimals=0)
    df_merged['10TH_DSO'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding(min_periods=1).quantile(0.1)).round(decimals=0)
    # df_merged['DSO_STD'] = df_merged.groupby(['obligor_code'])['days_outs_2m_avg'].transform(lambda x: x.expanding().std()).round(decimals=0)
    df_merged['AVG_DSO_LAST_6M'] = np.where(df_merged['AVG_DSO_LAST_6M'].isna(),df_merged['AVG_DSO'],df_merged['AVG_DSO_LAST_6M'])

    # Sort the rows
    df_merged = df_merged.sort_values(['obligor_code', 'invoice_yyyymm'])

    # New Flags
    df_merged['transaction_flag'] = np.where((df_merged['sales_flag']==1) | (df_merged['collections_flag']==1), 1, 0)
    df_merged['months_transacted'] = df_merged.groupby(['obligor_code'])['transaction_flag'].cumsum()
    df_merged['zero_current_balance'] = np.where(df_merged['balance']==0, 1, 0)
    df_merged['balance_lt_sales_prev_2m'] = np.where(df_merged['balance']<df_merged['sales_2m'], 1, 0)
    df_merged['balance_lt_sales_prev_3m'] = np.where(df_merged['balance']<df_merged['sales_3m'], 1, 0)
    df_merged['balance_lt_sales_prev_6m'] = np.where(df_merged['balance']<df_merged['sales_6m'], 1, 0)
    df_merged['likely_zero_balance'] = np.where(df_merged['balance'] < 8*1e5, 1, np.where(df_merged['paid_in_30_3m_avg'] >0.94, 1, 0))
    df_merged['dso_lt_100_T'] = np.where(df_merged['days_outs_2m_avg']<100, 1, 0)
    df_merged = get_exhaustive_dso(df_merged)
    df_merged['exhaustive_dso'] = df_merged['exhaustive_dso'].astype(float)

    # Target
    df_merged['dso_next_2m'] = df_merged.groupby(['obligor_code'])['exhaustive_dso'].transform(lambda x: x.shift(-2)).round(decimals=0)

    # Remove useless variables
    future_variables = ['balance_2m', 'balance_m_imputed', 'balance_2m_imputed', 'avg_invoice_amount_m_imputed', 
                        'avg_invoice_amount_2m_imputed', 'month', 'group', 'avg_invoice_amount_2m', 'next_2m_sales']
    df_merged.drop(future_variables, axis = 1, inplace = True)

    # Reset the Index
    df_merged = df_merged.reset_index(drop=True)
    logger.debug("\n\nVariables Created")

    return df_merged

# Create Build Sample & Outsample
def create_build_outsample(df_merged):
    # Build Sample Creation
    df_build = df_merged[(df_merged['invoice_yyyymm'] >= 202101) & (df_merged['invoice_yyyymm'] <= invoice_yyyymm[-3])]
    print("\n\nBuild Sample From 202101 to ", invoice_yyyymm[-3])
    print("Build Sample Shape: ", df_build.shape) 
    print("Build Sample Obligor Count: ", df_build.obligor_code.nunique()) 
    logger.debug("\n\nBuild Sample From 201901 to %s", invoice_yyyymm[-3])
    logger.debug("Build Sample Shape: %s", df_build.shape)
    logger.debug("Build Sample Obligor Count: %s", df_build.obligor_code.nunique())

    # Outsample Creation
    df_outsample = df_merged[(df_merged['invoice_yyyymm'] >= invoice_yyyymm[-2]) & (df_merged['invoice_yyyymm'] <= invoice_yyyymm[-1])]
    df_outsample = get_obligors_to_score(df_outsample, scored_obligors)
    print("\n\nOut Sample From ", invoice_yyyymm[-2], " to ", invoice_yyyymm[-1])
    print("Out Sample Shape: ", df_outsample.shape)
    print("Out Sample Obligor Count: ", df_outsample.obligor_code.nunique())
    logger.debug("\n\nOut Sample From %s to %s", invoice_yyyymm[-2], invoice_yyyymm[-1])
    logger.debug("Out Sample Shape: %s", df_outsample.shape)
    logger.debug("Out Sample Obligor Count: %s", df_outsample.obligor_code.nunique())

    return df_build, df_outsample

# Exclusions Before Model Building
def exclusions_before_model_building(df_merged):
    print("\n\nBefore All Exclusion: ", df_merged.shape)
    print("Obligor Count Before All Exclusion: ", df_merged.obligor_code.nunique())

    # Gets the obligor snap Count
    df_snap = df_merged[df_merged['beyond_max_snap'] == 0].groupby(['obligor_code'])['obligor_code'].count().to_frame(name='snap_count').reset_index(level=['obligor_code'])

    # Adding Snap count to the Dataframe
    df_merged = pd.merge(df_merged, df_snap, how='left', left_on=['obligor_code'], right_on = ['obligor_code'])

    # Creation of variable, more_than_6_snaps  which indicates whether an Obligor has more than been present for more than 6 months
    df_merged['more_than_6_snaps'] = np.where(df_merged['snap_count'] >= 6, 1, 0)

    # EXCLUSIONS TO CONSIDER, IF NECESSARY
    # 1 - Remove first 6 months of data of all customers and Remove customers with only 6 months of information
    # df_merged = df_merged[df_merged['obligor_age'] > 150]
    df_merged = df_merged[df_merged['more_than_6_snaps'] == 1]
    print("\nAfter Removing first 6 months of data of all customers: ", df_merged.shape)
    print("Obligor Count After Removing first 6 months of data of all customers: ", df_merged.obligor_code.nunique())
    logger.debug("\nAfter Removing first 6 months of data of all customers: %s", df_merged.shape)
    logger.debug("Obligor Count After Removing first 6 months of data of all customers: %s", df_merged.obligor_code.nunique())

    # 2 - Remove Obligors who are inactive (No sales & No Collections in last 6 Months)
    df_merged = df_merged[df_merged['no_sales_no_collection_last_6m'] == 0]
    print("\nAfter Removing customers who have no sales & no collections in last 6 months: ", df_merged.shape)
    print("Obligor Count After Removing customers who have no sales & no collections in last 6 months: ", df_merged.obligor_code.nunique())
    logger.debug("\nAfter Removing customers who have no sales & no collections in last 6 months: %s", df_merged.shape)
    logger.debug("Obligor Count After Removing customers who have no sales & no collections in last 6 months: %s", df_merged.obligor_code.nunique())

    # 3 - Removing all rows which has a NAN value
    null_variables = df_merged.columns[df_merged.isnull().any()]
    print("\nVariables with Null Values: ", null_variables)
    logger.debug("\nVariables with Null Values: %s", null_variables)
    print("Number of NULL values in the above variables \n", df_merged[null_variables].isnull().sum())

    before = df_merged.shape[0]
    print("Shape before Removing Rows with Nans: ", df_merged.shape)
    print("Obligor Count Shape before Removing Rows with Nans: ", df_merged.obligor_code.nunique())
    logger.debug("Shape before Removing Rows with Nans: %s", df_merged.shape)
    logger.debug("Obligor Count Shape before Removing Rows with Nans: %s", df_merged.obligor_code.nunique())
    df_merged.dropna(inplace=True)
    print("Shape after Removing Rows with Nans: ", df_merged.shape)
    print("Obligor Count Shape after Removing Rows with Nans: ", df_merged.obligor_code.nunique())
    logger.debug("Shape after Removing Rows with Nans: %s", df_merged.shape)
    logger.debug("Obligor Count Shape after Removing Rows with Nans: %s", df_merged.obligor_code.nunique())
    print("Number of rows dropped: ", before - df_merged.shape[0])
    logger.debug("Number of rows dropped: %s", before - df_merged.shape[0])

    # Remove temp variables
    temp = ['more_than_6_snaps', 'snap_count', 'beyond_max_snap', 'no_sales_no_collection_last_6m']
    df_merged.drop(temp, axis = 1, inplace = True)

    df_merged = df_merged.reset_index(drop=True)

    return df_merged

# Create Train & Test Sample
def create_train_test_sample(df_build, scored_obligors):
    invoice_yyyymm = pd.Series(df_build['invoice_yyyymm'].unique()).sort_values().to_list()
    # Train Sample Creation
    df_train = df_build[(df_build['invoice_yyyymm'] >= 202101) & (df_build['invoice_yyyymm'] < invoice_yyyymm[-6])]
    print("\n\nTrain Sample From 202101 to ", invoice_yyyymm[-5])
    print("Train Sample Shape: ", df_train.shape)   
    print("Train Sample Obligor Count: ", df_train.obligor_code.nunique())   
    logger.debug("\n\nTrain Sample From 202101 to %s", invoice_yyyymm[-5])
    logger.debug("Train Sample Shape: %s", df_train.shape) 
    logger.debug("Train Sample Obligor Count: %s", df_train.obligor_code.nunique()) 

    # Test Sample Creation
    df_test = df_build[(df_build['invoice_yyyymm'] >= invoice_yyyymm[-6]) & (df_build['invoice_yyyymm'] <= invoice_yyyymm[-3])]
    df_test = get_obligors_to_score(df_test, scored_obligors)
    print("\n\nTest Sample From ", invoice_yyyymm[-6], " to ", invoice_yyyymm[-3])
    print("Test Sample Shape: ", df_test.shape)
    print("Test Sample Obligor Count: ", df_test.obligor_code.nunique())
    logger.debug("\n\nTest Sample From %s to %s", invoice_yyyymm[-6], invoice_yyyymm[-3])
    logger.debug("Test Sample Shape: %s", df_test.shape) 
    logger.debug("Test Sample Obligor Count: %s", df_test.obligor_code.nunique()) 

    return df_train, df_test

# Save all the Files
def save_files(df_build, df_train, df_test, df_outsample, df_merged):
    # Resetting Indexes of Data Samples
    df_merged = df_merged.reset_index(drop=True)
    df_build = df_build.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Saving Build Sample, Train Sample, Test Sample and Out sample
    df_build.to_csv(build_sample, index=False)
    df_train.to_csv(train_sample, index=False)
    df_test.to_csv(test_sample, index=False)
    df_outsample.to_csv(outsample_file, index=False)
    df_merged.to_csv(whole_data, index=False)
    logger.debug("\n\nBuild Sample, Train Sample, Test Sample & Outsample Saved")

    return

def get_obligors_to_score(df, scored_obligors):
    df['obligor_code'] = df['obligor_code'].astype(str)
    df = df[df['obligor_code'].isin(scored_obligors)]
    df = df.sort_values(['obligor_code', 'invoice_yyyymm'])
    df.reset_index(drop=True, inplace=True)
    df['obligor_code'] = df['obligor_code'].astype(int)
    return df

if __name__ == "__main__":
    # Load Input Data & Preprocess
    df_merged, invoice_yyyymm, scored_obligors = preprocessing()

    # Exclusions and Imputations Before Variable Creation
    df_merged = exclusions_imputations_before_variable_creation(df_merged)

    # Variable Creation
    df_merged = create_new_variables(df_merged)

    # Create Build Sample & Outsample
    df_build, df_outsample = create_build_outsample(df_merged)

    # Exclusions Before Model Building
    df_build = exclusions_before_model_building(df_build)
    logger.debug("\n\nExclusions done on the Build Sample")

    # Create Train & Test Sample
    df_train, df_test = create_train_test_sample(df_build, scored_obligors)

    # Save all the Files
    save_files(df_build, df_train, df_test, df_outsample, df_merged)

    print("\n\nCode Completed")