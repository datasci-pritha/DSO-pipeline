import h2o
import os
import time
import pandas as pd
import numpy as np
import pdb
import warnings
warnings.filterwarnings('ignore')

from h2o.estimators.random_forest import H2ORandomForestEstimator

from bokeh.layouts import gridplot
from bokeh.plotting import figure, save

from config_connection_varirables import logger

# Input Data
input_file = 'dso_model/data/7_post_transformation/train_sample.csv'

# Uncomment if Remove MultiCollinearity is run before variable selection
# variable_file = 'dso_model/data/5_variable_shortlisting/variable_file.csv'

# Uncomment if Remove MultiCollinearity is not run before variable selection
variable_file = 'dso_model/data/5_variable_shortlisting/variable_input_file.csv'

# Output Data
out_var_importance = 'dso_model/data/6_random_forest/var_importance_'
conf_matrix = 'dso_model/data/6_random_forest/confusion_matrix.csv'
sco_hist = 'dso_model/data/6_random_forest/score_history_'
pdp_val = 'dso_model/data/6_random_forest/pdp/'
pdp_spline ='dso_model/data/6_random_forest/pdp_'

# Output Plots
rel_err_graph = 'dso_model/plots/2_random_forest/relative_error_graph_'
pdpplot = 'dso_model/plots/2_random_forest/pdp_plot/'
var_imp = 'dso_model/plots/2_random_forest/variable_importance_'
roc_plot = 'dso_model/plots/2_random_forest/roc_plot_'

target = 'dso_next_2m'

TOOLTIPS = [
	("index", "$index"),
    ("(x, y)", "($x, $y)")
]

def get_features():
    df = pd.read_csv(variable_file)

    # Get the Features
    features = df['NAME'][df['TYPE'] == 'PREDICTOR'].to_list()

    return features

def make_roc_plot(performance, var_seg):
    tpr = performance.tprs
    fpr = performance.fprs
    title = "ROC Curve -- AUC = " + str(round(performance.auc(), 2)) + " -- AUCPR = " + str(round(performance.pr_auc(), 2))
    # text = "AUCPR : " + str(round(perf.pr_auc(), 2))
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.1]
    p = figure(title=title, tools='', background_fill_color="#fafafa", tooltips=TOOLTIPS)
    p.line(x = fpr, y = tpr, line_width=4, line_color='#0f4c75')
    p.xaxis[0].ticker = ticks
    p.yaxis[0].ticker = ticks
    p.xaxis.axis_label = "False Positive Rate"
    p.yaxis.axis_label = "True Positive Rate"
    roc_plot_out = roc_plot + var_seg[0] + '.html'
    save(p, filename= roc_plot_out)

def variable_importance(inst):
    r = inst
    title = "Variable Importance"

    # Get the Variables
    variables = r._model_json['output']['variable_importances']['variable']

    # Get the Index of the Random Variable
    ran = variables.index('RANDOM')

    # Get the Scaeld & Relative Importance of the Variables
    scaled_importance = r._model_json['output']['variable_importances']['scaled_importance']
    relative_importance = r._model_json['output']['variable_importances']['relative_importance']

    if ran != 1:
        var = variables[0:ran-1]
        scaled_importance = scaled_importance[0:ran-1]
        relative_importance = relative_importance[0:ran-1] 
    else: 
        var = [variables[0]]
        scaled_importance = [scaled_importance[0]]
        relative_importance = [relative_importance[0]]

    # Revese all the list (To get the list in descending order of importance)
    var.reverse()
    scaled_importance.reverse()
    relative_importance.reverse()
    y_pos = np.arange(len(var))

    # Make the Vaiable Importance Graph
    p1 = make_variable_importance_plot(title, y = var, y_val = y_pos, x_val = scaled_importance, x_axis = "Scaled Importance")
    p2 = make_variable_importance_plot(title, y = var, y_val = y_pos, x_val = relative_importance, x_axis = "Relative Importance")

    # Stack the PLots Row Level
    plots_row = [p1, p2]

    # Make the Grid Plot
    grid = gridplot(plots_row, ncols=2, width=400, height=400)

    # Get the Output File Name
    var_imp_out = var_imp + '.html'

    # Save the Plot
    save(grid, filename= var_imp_out)

    return

def make_variable_importance_plot(title, y, y_val, x_val, x_axis):
    # Initialize the Plot
    p = figure(title=title, y_range= y, tools='', background_fill_color="#fafafa", tooltips=TOOLTIPS)

    # Add the horizontal histogram
    p.hbar(y =y, height = 0.5, left = 0, right = x_val, color = "#0f4c75")

    # Add the Axis Start & End
    p.x_range.start = 0
    p.x_range.end = round(max(x_val) * 1.2, 0)

    # Add the Axis Label
    p.xaxis.axis_label = x_axis

    # Change Grid Line Color to White
    p.grid.grid_line_color="white"

    return p

def relative_error_graph(rf, var_seg):
    #   Change Error Metric according to the model (i.e Linear or Logisitic Model)
    #   Regression : Use RMSE, MAE, Deviance
    #   Classification : Use Logloss, RMSE
    title = "Relative Error Graph"
    x_axis_label = "Number Of Trees"
    score_hist = rf._model_json['output']['scoring_history'].as_data_frame()
    score_hist = score_hist.drop(axis = 0, index = 0)
    score_hist1 = score_hist
    training_rmse = score_hist['training_rmse']
    validation_rmse = score_hist['validation_rmse']
    training_mse = [i**2 for i in training_rmse]
    validation_mse = [i**2 for i in validation_rmse]
    p1 = plot_relative_error_graph(title= title, x_val = score_hist['number_of_trees'], y_val_1 = training_mse, y_val_2 = validation_mse, x_axis_label = x_axis_label, y_axis_label = "MSE")
    p2 = plot_relative_error_graph(title= title, x_val = score_hist1['number_of_trees'], y_val_1 = score_hist1['training_mae'], y_val_2 = score_hist1['validation_mae'], x_axis_label = x_axis_label, y_axis_label = "MAE")

    plots_row = [p1, p2]
    grid = gridplot(plots_row, ncols=2, width=800, height=800)
    rel_err_graph_out = rel_err_graph + var_seg[0] + '.html'
    save(grid, filename= rel_err_graph_out)

def plot_relative_error_graph(title, x_val, y_val_1, y_val_2, x_axis_label, y_axis_label):
    p = figure(title=title, tools='', background_fill_color="#fafafa", tooltips=TOOLTIPS)

    p.line(x_val, y_val_1, line_width=4, line_color='firebrick', legend='Training')
    p.line(x_val, y_val_2, line_width=4, line_color='navy', legend='Validation')

    max_y = max(max(y_val_1), max(y_val_2))
    
    p.y_range.end = max_y
    p.legend.location = "top_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = x_axis_label
    p.yaxis.axis_label = y_axis_label
    p.grid.grid_line_color="white"

    return p

def pdp_plot(x, y, x_axis, y_axis):
    title = y_axis + " Vs " + x_axis 
    p = figure(title=title, tools='', background_fill_color="#fafafa", tooltips=TOOLTIPS)
    p.line(x = x, y = y, line_width=2)
    p.x_range.start = 0
    p.xaxis.axis_label = x_axis
    p.yaxis.axis_label = y_axis
    p.grid.grid_line_color="grey"

    return p

def pdp_values(rf, df, variables, var_seg):
    df_pdp = pd.DataFrame()
    variable = []
    mean_res = []
    stddev_res = []
    std_error_mean_resp =[]

    for var in variables:
        pdp_ = rf.partial_plot(data = df, cols = [var])
        variable.append(pdp_[0][var])
        mean_res.append(pdp_[0]["mean_response"])
        stddev_res.append(pdp_[0]["stddev_response"])
        std_error_mean_resp.append(pdp_[0]["std_error_mean_response"])

    df_pdp['variable'] = variable
    df_pdp['mean_response'] = mean_res
    df_pdp['stddev_response'] = stddev_res
    df_pdp['std_error_mean_response'] = std_error_mean_resp
    pdp_spline_out = pdp_spline+ var_seg[0] + '.csv'
    df_pdp.to_csv(pdp_spline_out)

    print("PDP Splines saved to file")

    return True

if __name__ == "__main__":
    # Load the Input Files
    df_ = pd.read_csv(input_file)

    # Get the Features
    feature_list = get_features()
    
    # Set the seed
    np.random.seed(666)

    # Create a Random Variable
    df_['RANDOM'] = np.random.rand(df_.shape[0])

    # Make a list of features and the target variable 
    cols = feature_list + ['RANDOM'] + [target]

    # Sort the DataFrame based on the Feature list
    df_ = df_[cols]

    # Initialize the H20 Instance & Remove all Instances
    h2o.init()
    h2o.remove_all()

    # Create a H20 DataFrame
    df = h2o.H2OFrame(df_)

    # Split the DataFrame into Train & Test
    train, valid = df.split_frame([0.8], seed=1234)

    # Get the Features & Target
    X = df.col_names[:-1] 
    y = df.col_names[-1]

    t1 = time.time()
    # RF model
    rf_v1 = H2ORandomForestEstimator(
        model_id="DSO Model",
        nfolds=10,                            # Number of folds for K-fold cross-validation (0 to disable or >= 2).
        ntrees=200,                           # Number of trees. Defaults to 50.
        max_depth=6,                          # Maximum tree depth (0 for unlimited). Defaults to 20.
        min_rows=3,                           # Fewest allowed (weighted) observations in a leaf. Defaults to 1.
        stopping_metric='MSE',                # Metric to use for early stopping (AUTO: logloss for classification, deviance for regression).
        stopping_rounds=10,                   # Early stopping based on convergence of stopping_metric. Common Metrics used are auto, mse, rmse, mae, rmsle, auc, aucpr, misclassification.
        stopping_tolerance=0.01,              # Relative tolerance for metric-based stopping criterion
        seed=666
        )
    
    print("Time Taken to Create the Model : ")
    print( time.time() - t1)
    t2 = time.time()

    # Training the model
    rf_v1.train(X, y, training_frame=train, validation_frame=valid)

    print("Time taken to Train the model : ")
    logger.debug("Finished Training the model")
    print(time.time() - t2)

    # Variable Importance
    variable_importance(inst = rf_v1)
    var_importance_out = out_var_importance + '.csv'
    rf_v1._model_json['output']['variable_importances'].as_data_frame().to_csv(var_importance_out, index=False)
    logger.debug("Variable Importance Graphs Created")

    h2o.shutdown(prompt=False)
    print("Code Completed")
