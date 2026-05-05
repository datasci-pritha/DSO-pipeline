import pdb
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from config_connection_varirables import logger, save_intermediate_files
from dso_model.a2_flooring_capping import floor_cap
from dso_model.a3_normalization import normalize

# Input File
input_test_file = 'dso_model/data/2_combined/test_sample.csv'

# Output File
test_file = 'dso_model/data/7_post_transformation/test_sample.csv'

def preprocess_test_data(df_test):
	# Floor the Test Data Set
	df_test = floor_cap(df_test, 'TEST')
	logger.debug("Flooring and Capping Done on Test Sample")

	# Normalize the Test Data Set
	df_test = normalize(df_test, 'TEST')
	logger.debug("Normalization Done on Test Sample")

	print("Test Sample Shape: ", df_test.shape)

	# Save Intermediate Files
	if save_intermediate_files:
		df_test.to_csv(test_file,index=False)

	return df_test

if __name__ == "__main__":
	# Load the Data
	df_test = pd.read_csv(input_test_file)

	# Preprocess the Test Data
	df_test = preprocess_test_data(df_test)

	print("Code Completed")