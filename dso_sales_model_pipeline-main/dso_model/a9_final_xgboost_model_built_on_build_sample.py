import pdb
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pickle import dump
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

from dso_model.a2_flooring_capping import floor_cap
from dso_model.a3_normalization import normalize
from config_connection_varirables import logger, save_intermediate_files

# Input Data Files
input_file = 'dso_model/data/2_combined/build_sample.csv'
outsample_file = 'dso_model/data/2_combined/outsample.csv'
input_var_file = 'dso_model/data/7_post_transformation/variable_output_file.csv'

# Intermediary Output Files
build_sample_post_transformations = "dso_model/data/9_final_model/post_transformation_whole_data.csv"
out_sample_post_transformations = "dso_model/data/9_final_model/post_transformation_whole_data.csv"

# Output Data Files
output_build_sample = 'dso_model/data/9_final_model/build_sample_scored_output.csv'
output_out_sample = 'dso_model/data/9_final_model/out_sample_scored_output.csv'
output_all_sample = 'dso_model/data/9_final_model/scored_output.csv'
feature_importance_file = 'dso_model/model_dumps/final_dump/xgboost_feature_importance.csv'
error_metrics_file = 'dso_model/model_dumps/error_metrics.csv'

# Output Plot Files
scatter_plot = 'dso_model/plots/4_final_model_validation/final_scatter_plot_dso_next_2m_actual_vs_dso_next_2m_pred.png'
hist_plot = 'dso_model/plots/4_final_model_validation/final_histogram_dso_next_2m_actual_vs_dso_next_2m_pred.png'
residual_plot = 'dso_model/plots/4_final_model_validation/final_residual_plot.png'
qq_plot = 'dso_model/plots/4_final_model_validation/final_qq_plot.png'

# Target Variable
target = 'dso_next_2m'

# Load and Preprocess Data
def lock_n_load_data():
    # Get Feature List
    feat = get_list_features(input_var_file)
    feature_array = get_list_features(input_var_file)
    features = pd.Index(feature_array)
    required_fields = feat
    # features_we_need = ['obligor_code', 'invoice_yyyymm', 'balance', 'balance_2m', 'next_2m_sales']
    features_we_need = ['obligor_code', 'invoice_yyyymm']
    required_fields = required_fields + features_we_need
    required_fields.append(target)

    # Read Files
    df = pd.read_csv(input_file)
    df_out = pd.read_csv(outsample_file)

    # Check if the target field is NULL or NOT
    if df_out[target].isnull().sum() == df_out.shape[0]:
        df_out[target] = 0

    # Floor the Build Sample & Out Sample
    df_floor = floor_cap(df, 'BUILD')
    logger.debug("\n\nFlooring and Capping Done on BUILD Sample")
    df_out_floor = floor_cap(df_out, 'OUTSAMPLE')
    logger.debug("Flooring and Capping Done on OUTSAMPLE")

    # Normalize the Build Sample & Out Sample
    df_normalize = normalize(df_floor, 'BUILD')
    logger.debug("\nNormalization Done on BUILD Sample")
    df_out_normalize = normalize(df_out_floor, 'OUTSAMPLE')
    logger.debug("Normalization Done on OUTSAMPLE")

    # Save the Files for later use
    df_normalize.to_csv(build_sample_post_transformations, index=False)
    logger.debug("\nBuild Sample Saved post transformations")
    df_out_normalize.to_csv(out_sample_post_transformations, index=False)
    logger.debug("OUTSAMPLE Saved post transformations")
    
    # Take only required Fields
    df_ = df_normalize[required_fields]
    df_out_ = df_out_normalize[required_fields]

    # Adding Obligor Unique Random Number
    df_w_obligor_rand_num = add_obligor_unique_random_num(df_)
    df_out_w_obligor_rand_num = add_obligor_unique_random_num(df_out_)

    # Preprocessing Data
    df_null_treated = preprocess_data(df_w_obligor_rand_num, 'BUILD')
    df_out_null_treated = preprocess_data(df_out_w_obligor_rand_num, 'OOS')

    return df_null_treated, df_out_null_treated, features_we_need, df, df_out, features

def split_training_validation_data(df_data, training_size):
	df_data['data_type'] = 'None'
	df_data['data_type'] = np.where(np.random.uniform(0, 1, len(df_data)) <= training_size, 'Training', 'Validation')
	df_training, df_validation = df_data[df_data['data_type']=='Training'], df_data[df_data['data_type']=='Validation']
	return df_training, df_validation

def get_training_data(df_training, features, y_target_field): 
    x = df_training[features]
    y = df_training[y_target_field]
    return x, y

def add_obligor_unique_random_num(df):
    iter, obligors = pd.factorize(df.obligor_code)
    choices = np.random.random(size = obligors.size)
    choice_random = np.random.choice(choices, obligors.shape, False)
    df['obligor_unique_random_num']=choice_random[iter]
    return df

def preprocess_data(df_data, sample):
    print('\n\nNumber of Rows Before preprocessing the ', sample, ' Data: ', df_data.shape)
    print('Obligor Count Before preprocessing the ', sample,' Data: ', df_data.obligor_code.nunique())
    logger.debug("\n\nNumber of Rows Before preprocessing the %s Data: %s", sample, df_data.shape)
    logger.debug("Obligor Count Before preprocessing the %s Data: %s", sample, df_data.obligor_code.nunique())
    df_data_processed = df_data.copy()
    temp = df_data_processed.isnull().any(axis=1)
    indexes = temp[temp == False].index
    df_data_processed = df_data_processed.iloc[indexes]
    print('\nNumber of Rows after preprocessing the ', sample, ' Data: ', df_data_processed.shape)
    print('Obligor Count after preprocessing the ', sample,' Data: ', df_data_processed.obligor_code.nunique())
    logger.debug("\nNumber of Rows after preprocessing the %s Data: %s", sample, df_data_processed.shape)
    logger.debug("Obligor Count after preprocessing the %s Data: %s", sample, df_data_processed.obligor_code.nunique())
    return df_data_processed

def write_columns(df, filename):
	columns_index = ['NAME', 'TYPE']
	df_columns = pd.DataFrame(columns=columns_index)
	col_ind = 0
	for column in df.columns:
		df_columns.loc[col_ind, 'NAME'] = column
		df_columns.loc[col_ind, 'TYPE'] = 'NORMAL'
		col_ind = col_ind + 1
	df_columns.to_csv(filename)
	return True

def get_list_features(filename):
	df_vars = pd.read_csv(filename)
	df_pred_vars = df_vars.loc[df_vars['TYPE'] == 'PREDICTOR']
	predictors = df_pred_vars['NAME'].tolist()
	return predictors

def get_scored_dataset(X, y, y_pred):
	actual = target + "_actual"
	pred = target + "_pred"
	df_predicted = X.reindex(columns = np.append(X.columns.values, [actual, pred]))
	df_predicted.loc[:, [pred]] = y_pred
	df_predicted.loc[:, [actual]] = y
	return df_predicted

def get_scored_dataset_modified(X, y, y_pred):
	actual = target + "_actual"
	pred = target + "_pred"
	df_predicted = X.reindex(columns = np.append(X.columns.values, [actual, pred]))
	df_predicted[pred] = y_pred
	df_predicted[actual] = y
	return df_predicted

def get_r_squared(actual, pred):
    SSE = ((pred-actual)**2).sum()
    y_hat = actual.mean()
    SST = ((actual-y_hat)**2).sum()
    return(1-SSE/SST)

def train_model(X_train, y_train):
    xgb_reg = xgb.XGBRegressor(max_depth=4, 
                                n_estimators=1100, 
                                colsample_bytree=0.7,
                                subsample=0.8,
                                n_jobs=2,
                                objectvie='reg:squarederror', 
                                booster='gbtree',
                                random_state=42, 
                                learning_rate=0.02)

    # Train the model with train data sets
    xgb_reg.fit(X_train, y_train)

    # Feature Importance
    fi = xgb_reg.get_booster().get_score(importance_type='gain')
    df_feat = pd.DataFrame(fi.items(), columns=['Features', 'Gain'])
    print("Feature Importances: \n", fi)

    # Compute VIF
    vif = compute_vif(X_train, X_train.columns.to_list())

    # Merge the VIF with Gain
    feature_important = pd.merge(df_feat, vif, left_on='Features', right_on='Variable', how='inner')
    feature_important.drop(['Variable'], axis = 1, inplace = True)
    feature_important.sort_values(['Gain'], inplace=True)

    # Save the Feature Importance With VIF
    feature_important.to_csv(feature_importance_file, index=False)

    # Save the Model
    dump(xgb_reg, open('dso_model/model_dumps/final_dump/xgboost.pkl', 'wb'))
    logger.debug("Saved the Model")

    return xgb_reg

def floor_cap_target_n_save_files(df, df_out, features_we_need, X, y, y_pred, X_out, y_out, y_out_pred):
    df_full = df[features_we_need]
    X_1 = X.join(df_full)
    df_pred = get_scored_dataset_modified(X_1, y, y_pred)

    # Floor and Cap the Buidl Sample
    df_pred['dso_next_2m_actual'] = np.where(df_pred['dso_next_2m_actual'] < 0, 0, df_pred['dso_next_2m_actual'])
    df_pred['dso_next_2m_actual'] = np.where(df_pred['dso_next_2m_actual'] > 350, 350, df_pred['dso_next_2m_actual'])
    df_pred['dso_next_2m_pred'] = np.where(df_pred['dso_next_2m_pred'] < 0, 0, df_pred['dso_next_2m_pred'])
    df_pred['dso_next_2m_pred'] = np.where(df_pred['dso_next_2m_pred'] > 350, 350, df_pred['dso_next_2m_pred'])

    df_full_out = df_out[features_we_need]
    X_out_1 = X_out.join(df_full_out)
    df_out_pred = get_scored_dataset_modified(X_out_1, y_out, y_out_pred)

    # Floor and Cap the Out Sample
    df_out_pred['dso_next_2m_actual'] = 0
    df_out_pred['dso_next_2m_pred'] = np.where(df_out_pred['dso_next_2m_pred'] < 0, 0, df_out_pred['dso_next_2m_pred'])
    df_out_pred['dso_next_2m_pred'] = np.where(df_out_pred['dso_next_2m_pred'] > 350, 350, df_out_pred['dso_next_2m_pred'])

    # Combine Build and Out
    df_all = pd.concat([df_pred, df_out_pred], axis=0, sort=True) 

    # Saving the output file
    df_pred.to_csv(output_build_sample, index=False)
    df_out_pred.to_csv(output_out_sample, index=False)
    df_all.to_csv(output_all_sample, index=False)
    logger.debug("Saved the Predicted Values and the Files")

    return df_pred, df_out_pred, df_all

def create_all_plots(df_pred):
    # Scatter Plot - dso_next_2m_actual vs dso_next_2m_pred
    plt.clf()
    plt.scatter(df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_pred'], c='orange')
    plt.plot(df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_actual'], c='black')
    plt.legend(loc='upper right')
    plt.savefig(scatter_plot)
    logger.debug("Saved the Scatter Plot - dso_next_2m_actual vs dso_next_2m_pred")

    # Histogram - dso_next_2m_actual vs dso_next_2m_pred
    plt.clf()
    bins = np.linspace(0, 350, 20)
    plt.hist([df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_pred']], bins, label=['actual', 'Model'])
    plt.legend(loc='upper right')
    plt.savefig(hist_plot)
    logger.debug("Saved the Histogram - dso_next_2m_actual vs dso_next_2m_pred")

    # Residual Plot - dso_next_2m_actual vs dso_next_2m_pred
    plt.clf()
    res = df_pred['dso_next_2m_actual'] - df_pred['dso_next_2m_pred']
    plt.scatter(res.index, res)
    plt.savefig(residual_plot)

    # # QQ Plot - Residual
    # plt.clf()
    # graphics.gofplots.qqplot(res, line='r')
    # plt.savefig(qq_plot)

    return

def get_error_metrics_b4_flooring(y_train, y_pred_train):
    # Load the Error File 
    df_error = pd.read_csv(error_metrics_file)

    # RMSE, R Squared on Train Data
    MSE_train = mse(y_train, y_pred_train)
    RMSE_train = np.sqrt(MSE_train)
    R_squared_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    # Append to the Error Metrics
    error_train = {'segment':'Before Capping', 'dataset':'Build', 'rmse':RMSE_train, 'r_squared':R_squared_train, 'mae':mae_train}
    df_error = df_error._append(error_train, ignore_index=True)

    print("\nRMSE on Build Sample: ", np.round(RMSE_train, 2))
    print("R-Squared on Build Sample: ", np.round(R_squared_train, 2))

    return df_error

def get_error_metrics(df_pred, df_error):
    # RMSE
    rmse_build = mse(df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_pred'], squared = False)
    print("RMSE of Build Sample: ", rmse_build)
    logger.debug("\n\n\n RMSE of Training Sample: %s \n\n\n ", rmse_build)

    # R Squared
    r_sq_build = get_r_squared(df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_pred'])
    print("R Squared of Build Sample: ", r_sq_build)
    logger.debug("\n\n\n R Squared of Build Sample: %s \n\n\n ", r_sq_build)

    # MAE
    mae_build = mean_absolute_error(df_pred['dso_next_2m_actual'], df_pred['dso_next_2m_pred'])
    print("MAE of Build Sample: ", mae_build)
    logger.debug("\n\n\n MAE of Build Sample: %s \n\n\n ", mae_build)

    # Append to the Error Metrics
    error_train = {'segment':'After Capping', 'dataset':'Build', 'rmse':rmse_build, 'r_squared':r_sq_build, 'mae':mae_build}
    df_error = df_error._append(error_train, ignore_index=True)

    if save_intermediate_files:
        df_error.to_csv(error_metrics_file, index=False)

    return

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

if __name__ == "__main__":
    # Load and Preprocess Data
    df_null_treated, df_out_null_treated, features_we_need, df, df_out, features = lock_n_load_data()

    # Sample Splitting
    X, y = get_training_data(df_null_treated, features, target)
    X_out, y_out = get_training_data(df_out_null_treated, features, target)

    print("\n\nAfter Null Treatment Build Sample Shape: ", X.shape)
    print("After Null Treatment Out Sample Shape: ", X_out.shape)

    # Train the Model
    xgb_reg = train_model(X, y)

    # Prediction
    y_pred = xgb_reg.predict(X)
    y_out_pred = xgb_reg.predict(X_out)

    # Get Error Metrics Before Flooring
    df_error = get_error_metrics_b4_flooring(y, y_pred)

    # Floor and Cap the Target and Save the Files
    df_pred, df_out_pred, df_all = floor_cap_target_n_save_files(df, df_out, features_we_need, X, y, y_pred, X_out, y_out, y_out_pred)

    # Create All Plots
    create_all_plots(df_pred)

    # Get the Error Metrics
    get_error_metrics(df_pred, df_error)

    print("Code Completed")