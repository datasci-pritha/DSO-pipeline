import pdb
import numpy as np
import pandas as pd
from pandas import ExcelWriter

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

from b4_remove_multicollinearity import compute_vif

# Input Files
build_sample_file = 'data/2_combined/build_sample.csv'
train_sample_file = 'data/7_post_transformation/train_sample.csv'
variable_file = 'data/7_post_transformation/variable_output_file.csv'

# Output Files
distribution_metrics_file = 'data/misc/distribtion_metrics.xls'
box_plot = 'plots/distributions/box_plot_'
hist_plot = 'plots/distributions/hist_plot_'
kernel_plot = 'plots/distributions/kernel_plot_'
cum_plot = 'plots/distributions/cum_plot_'

def load_data_preprocess():
    # Load the Variable Data
    df_var = pd.read_csv(variable_file)

    # Get the Variables
    variables = df_var['NAME'][df_var['TYPE']=='PREDICTOR'].to_list()

    # Columns Required
    cols_required = ['obligor_code', 'invoice_yyyymm'] + variables + ['dso_next_2m']

    # Load the Train Sample Data
    df = pd.read_csv(train_sample_file, usecols=cols_required)

    # Remove all [inf, nan, -inf]
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # Compute VIF
    vif = compute_vif(df, variables)

    # Get the Snap List
    snaps = pd.Series(df['invoice_yyyymm'].unique()).sort_values().to_list()

    # Seperate data into the last 6 Months Data and Remaining Data
    df_first = df[df['invoice_yyyymm'].isin(snaps[:-6])]
    df_last = df[df['invoice_yyyymm'].isin(snaps[-6:])]

    return variables, df_first, df_last, vif

def get_metrics(variables, df_first, df_last):
    # Create a DataFrame to Store the Distribution Metrics
    df_dist = pd.DataFrame(columns=['variable', 'ks_statistics', 'p_value'])

    for var in variables:
        if df_first.empty or df_last.empty:
            pdb.set_trace()
        # Get the KS Statistics
        temp = ks_2samp(df_first[var], df_last[var])
        data = {'variable': var, 'ks_statistics':temp[0], 'p_value':temp[1]}

        # Append to the Metrics DataFrame
        df_dist = df_dist.append(data, ignore_index=True)

        # Plot Scatter Plots
        plot_scatter_plot(var, df_first[var], df_last[var])

    return df_dist

def plot_scatter_plot(var, x, y):
    # Concatenate the Two Datasets
    df = pd.concat([x.to_frame().assign(dataset='Not Last 6 Months'), y.to_frame().assign(dataset='Last 6 Months')])

    # Update the plot path
    box_plot_path = box_plot + var + '_.png'
    hist_plot_path = hist_plot + var + '_.png'
    kernel_plot_path = kernel_plot + var + '_.png'
    cum_plot_path = cum_plot + var + '_.png'

    # Boxplot
    plt.clf()
    sns.boxplot(data=df, x='dataset', y=var)
    plt.title("Boxplot")
    plt.savefig(box_plot_path)

    # Histogram
    plt.clf()
    sns.histplot(data=df, x=var, hue='dataset', bins=50, stat='density', common_norm=False)
    plt.title("Density Histogram")
    plt.savefig(hist_plot_path)

    # Kernel Density Plot
    plt.clf()
    sns.kdeplot(x=var, data=df, hue='dataset', common_norm=False)
    plt.title("Kernel Density Function")
    plt.savefig(kernel_plot_path)

    # Cumulative Distribution Plot
    plt.clf()
    sns.histplot(x=var, data=df, hue='dataset', bins=len(df), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False)
    plt.title("Cumulative distribution function")
    plt.savefig(cum_plot_path)

    return

if __name__ == "__main__":
    # Load the Data, Preprocess, Get VIF & Variables, Split the Data
    variables, df_first, df_last, vif = load_data_preprocess()

    # Get the Distribution Metrics
    df_dist = get_metrics(variables, df_first, df_last)

    # Save the Distribution Metrics & VIF Values
    with ExcelWriter(distribution_metrics_file) as writer:
        df_dist.to_excel(writer,'KS_Staistics & P Value')
        vif.to_excel(writer,'VIF')

