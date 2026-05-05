##new code
from math import gamma
import pdb
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import ExcelWriter
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from statsmodels import graphics
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pickle import dump, load
from config_connection_varirables import conn, logger, save_intermediate_files

# Input Data Files
input_file_train_sample = 'sales_model/data/7_post_transformation/train_sample.csv'
input_file_test_sample = 'sales_model/data/7_post_transformation/test_sample.csv'
input_var_file = 'sales_model/data/7_post_transformation/variable_file.csv'

# Output Data Files
output_scored_build_training_sample = 'sales_model/data/8_scored_model/linear_pred_train.csv'
output_scored_test_sample = 'sales_model/data/8_scored_model/linear_pred_test.csv'
output_all_sample = 'sales_model/data/8_scored_model/combined_prediction.csv'
feature_importance_file_w_all_feats = 'sales_model/model_dumps/train_test_dumps/xgboost_feature_importance_w_all_features.csv'
feature_importance_file = 'sales_model/model_dumps/train_test_dumps/xgboost_feature_importance.csv'
Confusion_matrix_file = 'sales_model/model_dumps/train_test_dumps/confusion_matrix.xlsx'
error_metrics_file = 'sales_model/model_dumps/error_metrics.csv'
output_var_file = 'sales_model/data/7_post_transformation/variable_output_file.csv'

# Output Plot Files
scatter_plot = 'sales_model/plots/3_model_validation/scatter_plot_next_2m_sales_actual_vs_next_2m_sales_pred.png'
hist_plot = 'sales_model/plots/3_model_validation/histogram_next_2m_sales_actual_next_2m_sales_pred.png'
residual_plot = 'sales_model/plots/3_model_validation/residual_plot.png'
qq_plot = 'sales_model/plots/3_model_validation/qq_plot.png'

# Target Variable
target = 'next_2m_sales'

# To Hyperopt or GridSearch or or Randomsearch or None
cross_validate_method = 'None'

# Number of Trails
num_eval = 20

# Hyperopt Parameter Space Declaration 
param_hyperopt= {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'eval_metric': 'mae',
    'gamma': hp.uniform ('gamma', 0,9),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1500, 100)),
    'max_depth': scope.int(hp.quniform('max_depth', 2, 12, 1)),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
    'alpha': hp.uniform('alpha', 0.0, 1.0)
}

# Hyperparameter Space to be searched
param_space = {'objective':['reg:squarederror'],
              'booster':['gbtree'],
              'learning_rate': [0.1, 0.01, 0.02, 0.03], 
              'max_depth': [3,4,5,6, 7],
              'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
              'n_estimators': [400,600,800,1000,1200],
              "reg_alpha"   : [0.1,0.2,0.5,0.8,0.9,1],
              "reg_lambda"  : [2,3,5],
              "gamma"       : [1,2,4,6,8,9]
              }

# Load and Preprocess Data
def lock_n_load_data():
    # Get Feature List
    feat = get_list_features(input_var_file)
    feature_array = get_list_features(input_var_file)
    features = pd.Index(feature_array)
    required_fields = feat
    features_we_need = ['obligor_code', 'invoice_yyyymm', 'next_2m_sales']
    required_fields = required_fields + features_we_need

    # Load Input Data
    df_train_sample = pd.read_csv(input_file_train_sample, usecols=required_fields)
    df_test_sample = pd.read_csv(input_file_test_sample, usecols=required_fields)

    # Adding Obligor Unique Random Number
    df_w_obligor_rand_num_train = add_obligor_unique_random_num(df_train_sample)
    df_w_obligor_rand_num_test = add_obligor_unique_random_num(df_test_sample)

    # Preprocess the Data - Remove rows with the Null
    df_null_treated_train_sample = preprocess_data(df_w_obligor_rand_num_train, 'TRAIN')
    df_null_treated_test_sample = preprocess_data(df_w_obligor_rand_num_test, 'TEST')

    return df_null_treated_train_sample, df_null_treated_test_sample, features, df_train_sample, df_test_sample, features_we_need

def split_training_validation_data(df_data, training_size):
	df_data['data_type'] = 'None'
	df_data['data_type'] = np.where(np.random.uniform(0, 1, len(df_data)) <= training_size, 'Training', 'Validation')
	df_training, df_validation = df_data[df_data['data_type']=='Training'], df_data[df_data['data_type']=='Validation']
	return df_training, df_validation

def sample_splitting(df_training, features, y_target_field): 
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
    df_data_processed = df_data
    temp = df_data_processed.isnull().any(axis=1)
    indexes = temp[temp == False].index
    df_data_processed = df_data_processed.iloc[indexes]
    print('\nNumber of Rows after preprocessing the ', sample, ' Data: ', df_data.shape)
    print('Obligor Count after preprocessing the ', sample,' Data: ', df_data.obligor_code.nunique())
    logger.debug("\nNumber of Rows after preprocessing the %s Data: %s", sample, df_data.shape)
    logger.debug("Obligor Count after preprocessing the %s Data: %s", sample, df_data.obligor_code.nunique())
    return df_data_processed

def get_list_features(filename):
	df_vars = pd.read_csv(filename)
	df_pred_vars = df_vars.loc[df_vars['TYPE'] == 'PREDICTOR']
	predictors = df_pred_vars['NAME'].tolist()
	return predictors

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

# Rebuild Variable File
def build_variable_file(variable):
    df = pd.read_csv(input_file_train_sample)
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

    df_vars.to_csv(output_var_file)
    print('Variable file written')

    return

def randomsearch(X_train, y_train):
    xgb_model = xgb.XGBRegressor(random_state=30)

    grid_obj_xgb = RandomizedSearchCV(xgb_model,param_space, cv=5,n_iter=15,scoring='neg_mean_absolute_error',verbose=5,n_jobs=12)
    grid_obj_xgb.fit(X_train, y_train, verbose = 1)

    return grid_obj_xgb

def hyperopt(X_train, y_train, num_eval=100):
    
    start = time.time()
    
    def objective_function(params):
        rgb = xgb.XGBRegressor(**params)
        score = cross_val_score(rgb, X_train, y_train, cv=5, scoring='r2').mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_hyperopt, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
    loss = [x['result']['loss'] for x in trials.trials]

    best_param_values = [x for x in best_param.values()]

    # We can use either XGBoost or LGB
    rgb_best = xgb.XGBRegressor(objective='reg:squarederror',
                                eval_metric='mae',
                                alpha=best_param_values[0],
                                colsample_by_tree=best_param_values[1],
                                gamma=best_param_values[2],
                                learning_rate=best_param_values[3],
                                max_depth=int(best_param_values[4]),
                                n_estimators=int(best_param_values[5]))
                                  
    rgb_best.fit(X_train, y_train)
    
    print("\n\tResults")
    print("Score best parameters: ", min(loss)*-1)
    logger.debug("Score best parameters: {}".format(min(loss)*-1))
    print("Best parameters: ", best_param)
    logger.debug("Best parameters: {}".format(best_param))
    print("Time elapsed: ", time.time() - start)
    logger.debug("Time taken to Optimize Hyperparameters: {}".format(time.time() - start))
    print("Parameter combinations evaluated: ", num_eval)
    logger.debug("Parameter combinations evaluated: {}".format(num_eval))
    print("Feature Importances: \n", rgb_best.get_booster().get_score(importance_type='gain'))
    logger.debug("Feature Importances: {}".format(rgb_best.get_booster().get_score(importance_type='gain')))
    
    return rgb_best

def train_model(X_train, y_train):
    if cross_validate_method == 'Hyperopt':
        xgb_reg = hyperopt(X_train, y_train, num_eval)

        # Feature Importance
        fi = xgb_reg.get_booster().get_score(importance_type='gain')

    elif cross_validate_method == 'Randomsearch':
        xgb_reg = randomsearch(X_train, y_train)

        # Feature Importance
        fi = {'Features': X_train.columns, 'Gain': xgb_reg.best_estimator_.feature_importances_}
    else:
    # Instantiate an XGBoost object with hyperparameters
    # 1st Instance
        xgb_reg = xgb.XGBRegressor(max_depth=7,
                                    n_estimators=600,
                                    colsample_bytree=0.7,
                                    objectvie='reg:squarederror',
                                    booster='gbtree',
                                    random_state=42,
                                    learning_rate=0.01,
                                    reg_lambda=5,
                                    reg_alpha=0.8,
                                    gamma=8
                                    )

        # Train the model with train data sets
        xgb_reg.fit(X_train, y_train)

        # Feature Importance
        fi = xgb_reg.get_booster().get_score(importance_type='gain')

    #  Save the Feature Importance File
    print("Feature Importances: \n", fi)
    df_feature = pd.DataFrame(fi.items(), columns=['Features', 'Gain'])
    df_feature = df_feature.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    df_feature['prev_gain'] = df_feature['Gain'].shift(periods=1, fill_value=0)
    df_feature['perc_change'] = (df_feature['prev_gain'] - df_feature['Gain']) * 100 / df_feature['prev_gain']

    return xgb_reg, df_feature

def floor_cap_target_n_save_files(df_train_sample, df_test_sample, features_we_need, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test):
    df_train = df_train_sample[features_we_need]
    X_train_1 = X_train.join(df_train)
    df_pred_train = get_scored_dataset_modified(X_train_1, y_train, y_pred_train)

    # Floor & Cap the Values - Train Data
    df_pred_train['next_2m_sales_actual'] = np.where(df_pred_train['next_2m_sales_actual'] < 1000, 1000, df_pred_train['next_2m_sales_actual'])
    df_pred_train['next_2m_sales_actual'] = np.where(df_pred_train['next_2m_sales_actual'] > 1.71*1e7, 1.71*1e7, df_pred_train['next_2m_sales_actual'])
    df_pred_train['next_2m_sales_pred'] = np.where(df_pred_train['next_2m_sales_pred'] < 1000, 1000, df_pred_train['next_2m_sales_pred'])
    df_pred_train['next_2m_sales_pred'] = np.where(df_pred_train['next_2m_sales_pred'] > 1.71*1e7, 1.71*1e7, df_pred_train['next_2m_sales_pred'])

    df_test = df_test_sample[features_we_need]
    X_test_1 = X_test.join(df_test)
    df_pred_test = get_scored_dataset_modified(X_test_1, y_test, y_pred_test)

    # Floor & Cap the Values - Test Data
    df_pred_test['next_2m_sales_actual'] = np.where(df_pred_test['next_2m_sales_actual'] < 1000, 1000, df_pred_test['next_2m_sales_actual'])
    df_pred_test['next_2m_sales_actual'] = np.where(df_pred_test['next_2m_sales_actual'] > 1.71*1e7, 1.71*1e7, df_pred_test['next_2m_sales_actual'])
    df_pred_test['next_2m_sales_pred'] = np.where(df_pred_test['next_2m_sales_pred'] < 1000, 1000, df_pred_test['next_2m_sales_pred'])
    df_pred_test['next_2m_sales_pred'] = np.where(df_pred_test['next_2m_sales_pred'] > 1.71*1e7, 1.71*1e7, df_pred_test['next_2m_sales_pred'])

    # Combine Train and Test
    df_all = pd.concat([df_pred_train, df_pred_test], axis=0, sort=True) 

    if save_intermediate_files:
        # Saving the output files
        df_pred_train.to_csv(output_scored_build_training_sample, index=False)
        df_pred_test.to_csv(output_scored_test_sample, index=False)
        df_all.to_csv(output_all_sample, index=False)
        logger.debug("Saved the Predicted Values")

    return df_all, df_pred_train, df_pred_test

def create_all_plots(df_all, df_pred_train):
    # Scatter Plot - next_2m_sales_actual vs next_2m_sales_pred
    plt.clf()
    plt.scatter(df_all['next_2m_sales_actual'], df_all['next_2m_sales_pred'], c='orange')
    plt.plot(df_all['next_2m_sales_actual'], df_all['next_2m_sales_actual'], c='black')
    plt.legend(loc='upper right')
    plt.savefig(scatter_plot)
    logger.debug("Saved the Scatter Plot - next_2m_sales_actual vs next_2m_sales_pred")

    # Histogram - next_2m_sales_actual vs next_2m_sales_pred
    plt.clf()
    bins = np.linspace(0, 350, 20)
    plt.hist([df_pred_train['next_2m_sales_actual'], df_pred_train['next_2m_sales_pred']], bins, label=['actual', 'Model'])
    plt.legend(loc='upper right')
    plt.savefig(hist_plot)
    logger.debug("Saved the Histogram - next_2m_sales_actual vs next_2m_sales_pred")

    # Residual Plot - next_2m_sales_actual vs next_2m_sales_pred
    plt.clf()
    res = df_pred_train['next_2m_sales_actual'] - df_pred_train['next_2m_sales_pred']
    plt.scatter(res.index, res)
    plt.savefig(residual_plot)

    # QQ Plot - Residual
    # plt.clf()
    # residual = linear_model.resid_response
    # fig = sm.qqplot(res, line='45')
    # graphics.gofplots.qqplot(res, line='r')
    # plt.savefig(qq_plot)

    return

def get_error_metrics(df_pred_train, df_pred_test, df_error):
    # RMSE
    rmse_train = mse(df_pred_train['next_2m_sales_actual'], df_pred_train['next_2m_sales_pred'], squared = False)
    rmse_test = mse(df_pred_test['next_2m_sales_actual'], df_pred_test['next_2m_sales_pred'], squared = False)
    print("RMSE of Training Sample: ", rmse_train)
    print("RMSE of Out Sample: ", rmse_test)
    logger.debug("RMSE of Training Sample: %s", rmse_train)
    logger.debug("RMSE of Out Sample: %s", rmse_test)

    # R Squared
    r_sq_train = get_r_squared(df_pred_train['next_2m_sales_actual'], df_pred_train['next_2m_sales_pred'])
    r_sq_test = get_r_squared(df_pred_test['next_2m_sales_actual'], df_pred_test['next_2m_sales_pred'])

    print("R Squared of Training Sample: ", r_sq_train)
    print("R Squared of Out Sample: ", r_sq_test)
    logger.debug("R Squared of Training Sample: %s", r_sq_train)
    logger.debug("R Squared of Out Sample: %s", r_sq_test)

    # MAE
    mae_train = mean_absolute_error(df_pred_train['next_2m_sales_actual'], df_pred_train['next_2m_sales_pred'])
    mae_test = mean_absolute_error(df_pred_test['next_2m_sales_actual'], df_pred_test['next_2m_sales_pred'])

    print("MAE of Training Sample: ", mae_train)
    print("MAE of Out Sample: ", mae_test)
    logger.debug("MAE of Training Sample: %s", mae_train)
    logger.debug("MAE of Out Sample: %s", mae_test)

    # To Store The Data
    error_train = {'segment':'After Capping', 'dataset':'Train', 'rmse':rmse_train, 'r_squared':r_sq_train, 'mae':mae_train}
    error_test = {'segment':'After Capping', 'dataset':'Test', 'rmse':rmse_train, 'r_squared':r_sq_test, 'mae':mae_test}

    # Append to the Error Metrics
    # df_error = df_error.append(error_train, ignore_index=True)
    # df_error = df_error.append(error_test, ignore_index=True)
    df_error = pd.concat([df_error, pd.DataFrame([error_train])], ignore_index=True)
    df_error = pd.concat([df_error, pd.DataFrame([error_test])], ignore_index=True)


    if save_intermediate_files:
        df_error.to_csv(error_metrics_file, index=False)

    return

# Error Metrics of Model
def error(X_train, X_test, y_train, y_test, xgb_reg, df_error):
    # Prediction
    y_pred_train = xgb_reg.predict(X_train)
    y_pred_test = xgb_reg.predict(X_test)

    # RMSE, R Squared on Train Data
    MSE_train = round(mse(y_train, y_pred_train), 2)
    RMSE_train = round(np.sqrt(MSE_train), 2)
    R_squared_train = round(r2_score(y_train, y_pred_train), 2)
    mae_train = round(mean_absolute_error(y_train, y_pred_train), 2)

    # RMSE, R Squared on Test Data
    MSE_test = round(mse(y_test, y_pred_test), 2)
    RMSE_test = round(np.sqrt(MSE_test), 2)
    R_squared_test = round(r2_score(y_test, y_pred_test), 2)
    mae_test = round(mean_absolute_error(y_test, y_pred_test), 2)

    # print('\n', target, ': \n', y_train.describe())

    print("\n\n\nRMSE on Train Sample: ", RMSE_train)
    print("R-Squared on Train Sample: ", R_squared_train)
    logger.debug("\n\n\n RMSE of Train Sample: %s \n\n\n ", RMSE_train)
    logger.debug("\n\n\n R-Squared of Train Sample: %s \n\n\n ", R_squared_train)

    print("\nRMSE on Test Sample: ", RMSE_test)
    print("R-Squared on Test Sample: ", R_squared_test)
    logger.debug("\n\n\n RMSE of Test Sample: %s \n\n\n ", RMSE_test)
    logger.debug("\n\n\n R-Squared of Test Sample: %s \n\n\n ", R_squared_test)

    # Append to the Error Metrics
    error_train = {'segment':'Before Capping', 'dataset':'Train', 'rmse':RMSE_train, 'r_squared':R_squared_train, 'mae':mae_train}
    error_test = {'segment':'Before Capping', 'dataset':'Test', 'rmse':RMSE_test, 'r_squared':R_squared_test, 'mae':mae_test}
    # df_error = df_error.append(error_train, ignore_index=True)
    # df_error = df_error.append(error_test, ignore_index=True)
    df_error = pd.concat([df_error, pd.DataFrame([error_train])], ignore_index=True)
    df_error = pd.concat([df_error, pd.DataFrame([error_test])], ignore_index=True)

    return y_pred_train, y_pred_test, df_error, y_pred_train, y_pred_test, R_squared_test

def create_sales_matrix(test):
    # Create Segment
    test['obligor_code'] = test['obligor_code'].astype(str)
    test['segment'] = np.where(test['obligor_code'].str.slice(0,2).isin(['70']), 'IB', 
                            (np.where(test['obligor_code'].str.slice(0,2).isin(['40']), 'GOVT', 'TRADE')))
    actual = target + '_actual'
    predicted = target + '_pred'

    # Iterate over each snap
    with ExcelWriter(Confusion_matrix_file) as writer:
        for snap in test.invoice_yyyymm.unique().tolist():
            print('\nSnap: ', snap)
            # Get Each Snap Data
            df = test[test['invoice_yyyymm']==snap].reset_index()

            for seg in ['ALL', 'TRADE', 'GOVT', 'IB']:
                print('\nSegment: ', seg)
                # Create a DataFrame For Binning
                df_binned = pd.DataFrame()

                # Get the Segmented Data
                if seg != 'ALL':
                    df_seg = df[df['segment']==seg].reset_index(drop=True)
                else:
                    df_seg = df

                # Bin the Actual & Predicted
                bins=[0, 1e5, 5*1e5, 20*1e5, 2*1e7]
                for col in [actual, predicted]:
                    df_binned[col] = pd.cut(df_seg[col], bins=bins)

                # Make Confusion Matrix
                labels = df_binned[actual].unique().sort_values().astype(str).tolist()
                confusion = confusion_matrix(df_binned[actual].astype(str), df_binned[predicted].astype(str), labels=labels)
                print('\n Confusion Matrix Created for %s Segment Snap Month: %s' % (seg, snap))

                # Save it to the Excel
                # if save_intermediate_files:
                sheet_name = str(snap) + '_' + seg
                pd.DataFrame(confusion).to_excel(writer,sheet_name)

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

# Variable Selection using XGBoost
def var_selection_xgb(X_train, X_test, y_train, y_test):
    # To Initiate Variable Selection
    var_selection = True

    # Index of Feature
    ind = 1

    # Maximum R Squared
    max_r_sq = 1

    # Model Training & Variable Selection
    while var_selection:

        # Store Error Metrics
        df_error = pd.DataFrame(columns=['segment', 'dataset', 'rmse', 'r_squared', 'mae'])

        if ind == 1:
            # Train the Model
            xgb_reg, df_feat = train_model(X_train, y_train)
            df_feat_imp = df_feat.copy()

            # Fit the model & get the Error Metrics of the model
            y_pred_train, y_pred_test, df_error, y_pred_train, y_pred_test, R_squared_test = error(X_train, X_test, y_train, y_test, xgb_reg, df_error)
        else:
            # Train the Model
            xgb_reg, df_feat = train_model(X_train[feats], y_train)
            
            # Fit the model & get the Error Metrics of the model
            y_pred_train, y_pred_test, df_error, y_pred_train, y_pred_test, R_squared_test = error(X_train[feats], X_test[feats], y_train, y_test, xgb_reg, df_error)

        # Checking if the R Squared dropped significantly
        if ind == 1:
            max_r_sq = R_squared_test
            df_error_first = df_error.copy()
            df_error_first['segment'] = 'Before Capping W All Features'
        else:
            change = round((max_r_sq - R_squared_test) * 100, 1)
            print('\nChange in R Squared = ', change)

            if change < 3:
                var_selection = False
                
                # Update the Train & Test
                X_train = X_train[feats]
                X_test = X_test[feats]
                
                # Save the Feature Importance With VIF
                vif = compute_vif(X_train, feats)
                feature_important = pd.merge(df_feat[['Features', 'Gain']], vif, left_on='Features', right_on='Variable', how='inner')
                feature_important.sort_values(['Gain'], inplace=True)
                feature_important.to_csv(feature_importance_file)
                df_feat_imp.to_csv(feature_importance_file_w_all_feats, index=False)

                # Build the Variable File
                build_variable_file(feats)

                # Save the Model
                dump(xgb_reg, open('sales_model/model_dumps/train_test_dumps/xgboost.pkl', 'wb'))
                logger.debug("Saved the Model")

        # Get the index
        try:
            print('\nIndex: ', df_feat_imp['perc_change'])
            feat_index = df_feat_imp[df_feat_imp['perc_change'] < 100].index[ind]
        except:
            try:
                feat_index = df_feat_imp[df_feat_imp['perc_change'] < 150].index[ind]
            except Exception as e:
                print('Error: ', e)
                    



        # Get the Features to be Considered
        feats = df_feat_imp['Features'][:feat_index].to_list()

        # Features to be considered in the next Iteration
        print('\nFeatures to be considered in next Iteration: \n', feats)

        ind = ind + 1

    return y_pred_train, y_pred_test, df_error

if __name__ == "__main__":
    # Load and Preprocess Data
    df_null_treated_train_sample, df_null_treated_test_sample, features, df_train_sample, df_test_sample, features_we_need = lock_n_load_data()

    # Sample Splitting
    X_train, y_train = sample_splitting(df_null_treated_train_sample, features, target)
    X_test, y_test = sample_splitting(df_null_treated_test_sample, features, target)
    
    print("Training Sample Shape: ", X_train.shape)
    print("Test Sample Shape: ", X_test.shape)

    y_pred_train, y_pred_test, df_error = var_selection_xgb(X_train, X_test, y_train, y_test)
    
    # Floor and Cap the Target and Save the Files
    df_all, df_pred_train, df_pred_test = floor_cap_target_n_save_files(df_train_sample, df_test_sample, features_we_need, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test)

    # Create All Plots
    create_all_plots(df_all, df_pred_train)

    # Get the Error Metrics
    df_error = get_error_metrics(df_pred_train, df_pred_test, df_error)

    # Get the Confusion Matrix
    create_sales_matrix(df_pred_test)

    print("Code Completed")