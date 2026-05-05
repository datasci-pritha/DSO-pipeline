import pdb
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import Span
from bokeh.io import export_png

from config_connection_varirables import logger

# Input Files
input_file = 'dso_model/data/2_combined/train_sample.csv'

# Output Files
output_file = 'dso_model/data/3_flooring_capping/train_sample_post_outlier_trtmnt.csv'
stats_file = 'dso_model/data/3_flooring_capping/variable_stats.csv'
stats_after = 'dso_model/data/3_flooring_capping/stats_after_floor_cap.csv'
full_data_stats_file = 'dso_model/data/3_flooring_capping/build_sample_variable_stats.csv'

def floor_cap(df_wo_flooring_capping, sample):
    sales_bal_floor = 0
    sales_bal_cap = '95%'
    dso_floor = 0
    dso_cap = 360
    paid_floor = 0
    paid_cap = 1

    # Variable Segments
    sales_var = ['sales', 'sales_till_now', 'sum_invoices_3m_obl', 'sum_invoices_6m_obl', 'sales_3m', 'sales_2m', 'sales_6m', 
                'sales_1', 'sales_2', 'sales_3', 'SALES_3M', 'SALES_4M', 'SALES_6M', 'AVG_SALES_LAST_6M', 'AVG_SALES'
                'sales_2m_1', 'sales_2m_2', 'AVG_SALES_2M_LAST_6M', 'AVG_SALES_2M', 'MAX_SALES_2M', 'MIN_SALES_2M', '90TH_SALES_2M', '10TH_SALES_2M', 
                'SALES_2M_STD', 'sales_growth', 'AVG_SALES_GROWTH', 'MAX_SALES_GROWTH', 'MIN_SALES_GROWTH', '90TH_SALES_GROWTH', '10TH_SALES_GROWTH'
                'avg_invoice_amount', 'avg_invoice_amount_per_month']

    paid_var = ['paid_in_30_3m_avg', 'paid_in_60_3m_avg', 'paid_in_90_3m_avg', 'paid_in_31_60_3m_avg', 'paid_in_61_90_3m_avg', 'paid_in_30_6m_avg', 
                'paid_in_60_6m_avg', 'paid_in_90_6m_avg', 'paid_in_31_60_6m_avg', 'paid_gt_90_6m_avg', 'paid_ever_6m', 'paid_in_61_90_6m_avg', ]

    balance_var = ['balance', 'balance_1', 'balance_2', 'balance_3', 'balance_prev2m', 'balance_prev2m_1', 'AVG_Balance_LAST_6M', 
                    'AVG_Balance', 'MAX_Balance', 'MIN_Balance', '90TH_Balance', '10TH_Balance', 'Balance_STD']

    dso_var = ['days_outs_2m_avg', 'dso_prev2m_1', 'days_outs_6m_avg', 'days_outs_3m_avg', 'days_outs_4m_avg', 'AVG_DSO_LAST_6M', 'AVG_DSO', 'MAX_DSO', 
                'MIN_DSO', '90TH_DSO', '10TH_DSO', 'DSO_STD']

    # Consider only non categorical variables 
    df_non_catg_vars = df_wo_flooring_capping.select_dtypes(include=['float64', 'int64'])

    # Remove all binary variables 
    s_nunique = df_non_catg_vars.nunique()
    s_nunique = s_nunique[s_nunique > 2]
    col_list = s_nunique.index.tolist()
    # print(col_list)
    
    # Remove variables/columns that shouldn't be floored and capped
    columns_to_remove = ['obligor_code', 'invoice_yyyymm', 'invoices_per_month', 'avg_invoice_amount_2m', 'balance_2m', 'next_2m_sales', 'dso_next_2m',
                        'sales_active_last_m', 'sales_active_last_2m', 'sales_active_last_3m', 'paid_never_last_3m', 'paid_never_last_6m', 'paid_ever_6m', 
                        'trade_flag', 'govt_flag', 'ib_flag', 'cna_flag', 'invoices_per_month', 'yyyymm', 'overdue_amount',	'bp_number', 'credit_days',	
                        'sap_credit_limit', 'quarter_ending', 'month_month_1', 'month_month_2', 'month_month_3', 'month_month_4', 'month_month_5',
                        'month_month_6', 'month_month_7', 'month_month_8', 'month_month_9', 'month_month_10', 'month_month_11', 'month_month_12',
                        'transaction_flag', 'sales_flag', 'collections_flag', 'months_transacted', 'no_of_months_collections', 'no_of_months_sales'
                        ]

    for cols in columns_to_remove:
        try:
            col_list.remove(cols)
        except Exception:
            pass

    if sample == 'TRAIN':
        df_stats = df_wo_flooring_capping.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all')
        df_stats.to_csv(stats_file)
        logger.debug("Stats File Saved for Training Sample Saved")
    elif sample == 'TEST':
        df_stats = pd.read_csv('dso_model/data/3_flooring_capping/variable_stats.csv', index_col=0)
    elif sample == 'BUILD':
        df_stats = df_wo_flooring_capping.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all')
        df_stats.to_csv('dso_model/data/3_flooring_capping/build_sample_variable_stats.csv')
        logger.debug("Stats File Saved for Build Sample Saved")
    elif sample == 'OUTSAMPLE':
        df_stats = pd.read_csv('dso_model/data/3_flooring_capping/build_sample_variable_stats.csv', index_col=0)

    df_w_flooring_capping = df_wo_flooring_capping.copy()

    for column in col_list:
        if (column in sales_var) or (column in balance_var):
            row_index = 0
            for row in df_w_flooring_capping[column]:
                if row < sales_bal_floor:
                    df_w_flooring_capping.at[row_index, column] = sales_bal_floor
                elif row > df_stats.at[sales_bal_cap, column]:
                    df_w_flooring_capping.at[row_index, column] = df_stats.at[sales_bal_cap, column]
                row_index = row_index + 1 

        elif (column in dso_var):
            floor = dso_floor
            cap = dso_cap
            row_index = 0
            for row in df_w_flooring_capping[column]:
                if row < floor:
                    df_w_flooring_capping.at[row_index, column] = floor
                elif row > cap:
                    df_w_flooring_capping.at[row_index, column] = cap
                row_index = row_index + 1 

        elif (column in paid_var):
            floor = paid_floor
            cap = paid_cap
            row_index = 0
            for row in df_w_flooring_capping[column]:
                if row < floor:
                    df_w_flooring_capping.at[row_index, column] = floor
                elif row > cap:
                    df_w_flooring_capping.at[row_index, column] = cap
                row_index = row_index + 1             
        else:
            floor = '1%'
            cap = '99%'
            row_index = 0
            for row in df_w_flooring_capping[column]:
                if row < df_stats.at[floor, column]:
                    df_w_flooring_capping.at[row_index, column] = df_stats.at[floor, column]
                elif row > df_stats.at[cap, column]:
                    df_w_flooring_capping.at[row_index, column] = df_stats.at[cap, column]
                row_index = row_index + 1 

    if sample == 'TRAIN':
        return df_w_flooring_capping, col_list
    elif sample == 'BUILD':
        after_flooring = 'dso_model/data/3_flooring_capping/post_flooring_capping_build_sample_variable_stats.csv'
        df_w_flooring_capping.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all').to_csv(after_flooring)
        return df_w_flooring_capping
    else:
        return df_w_flooring_capping

def create_plots(df_wo_flooring_capping, df_w_flooring_capping, col_list):
    plots = []
    for column in col_list:
        try:
            row = []
            minx = math.floor(df_wo_flooring_capping[column].min())
            maxx = math.floor(df_wo_flooring_capping[column].max()) + 1
            x = np.linspace(minx, maxx, 1000)
            np_array_col_wo = df_wo_flooring_capping[column].values
            hist_wo, edges_wo = np.histogram(np_array_col_wo, density=True, bins=50)

            floor_marker = math.floor(df_w_flooring_capping[column].min())
            cap_marker = math.floor(df_w_flooring_capping[column].max())

            np_array_col_w = df_w_flooring_capping[column].values
            hist_w, edges_w = np.histogram(np_array_col_w, density=True, bins=50)

            p1 = make_plot(title='Without Flooring Capping', hist=hist_wo, edges=edges_wo, x=x, floor_marker=floor_marker, cap_marker=cap_marker, xaxis_label=column, yaxis_label='Num')
            p2 = make_plot(title='With Flooring Capping', hist=hist_w, edges=edges_w, x=x, floor_marker=floor_marker, cap_marker=cap_marker, xaxis_label=column, yaxis_label='Num')
            row.append(p1)
            row.append(p2)
            plots.append(row)
            #Uncomment this if you want pngs instead
            outfilename_png = "plots/1_univariate/"+column+".png"
            # outfilename_htm = "plots/1_univariate/"+column+".html"
            grid = gridplot([p1, p2], ncols=2, plot_width=800, plot_height=800, toolbar_location=None)
            export_png(grid, filename=outfilename_png)
            # save(grid, filename=outfilename_htm)
        except Exception:
            pass

def make_plot(title, hist, edges, x, floor_marker, cap_marker, xaxis_label, yaxis_label):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    vline1 = Span(location=floor_marker, dimension='height', line_color='#29ba74', line_width=3)
    vline2 = Span(location=cap_marker, dimension='height', line_color='red', line_width=3)

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label
    p.grid.grid_line_color="white"
    p.renderers.extend([vline1, vline2])
    return p

if __name__ == "__main__":
    # Load the Input File
    df_wo_flooring_capping = pd.read_csv(input_file)

    # Flooring and capping of Train Sample
    df_w_flooring_capping, col_list = floor_cap(df_wo_flooring_capping, 'TRAIN')
    logger.debug("Flooring and Capping Done on the Training Sample")

    # Save the Floored & Capped Data Frame
    df_w_flooring_capping.to_csv(output_file, index=False)
    df_w_flooring_capping.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all').to_csv(stats_after)

    # Plots the univarite graphs of the Variables
    create_plots(df_wo_flooring_capping, df_w_flooring_capping, col_list)
    logger.debug("Univariate Plots Created")
    print("Code Completed")