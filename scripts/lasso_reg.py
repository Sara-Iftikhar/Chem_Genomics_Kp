
import gc
import sys
import joblib
import pandas as pd
import numpy as np
import logging
from ai4water.utils.utils import TrainTestSplit
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from SeqMetrics import RegressionMetrics
from ai4water.utils.utils import dateandtime_now

from utils import prepare_data

jobid = sys.argv[1]
time_of_this_file = dateandtime_now()
file_type = sys.argv[2]
target_ID = int(sys.argv[3])

USE_TRAINED_MODEL = True

# Load predictor data
X, dependent = prepare_data(dependent_file_type=file_type)
target = dependent.columns[target_ID]

# Load dependent variable data
y = dependent.iloc[:, target_ID]

logging.basicConfig(filename=f'log_files/lasso/test_danesh_{dateandtime_now()}_{jobid}_{file_type}_{dependent.columns[target_ID]}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info(f"job ID: {jobid}")

logger.info(f'target: {dependent.columns[target_ID]}, file_type: {file_type}')


logger.info(f'X shape {X.shape}')
logger.info(X.head())

logger.info(f'y shape {dependent.shape}')
logger.info(dependent.columns)
logger.info(dependent.head())

# Prepare lists for results
interval_coverage_list = []
interval_width_list = []
range_value_list = []
heritability_list_train = []  # Store heritability from training dataset
heritability_list_test = []    # Store heritability from test dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = TrainTestSplit(seed=313).split_by_random(X, y)
logger.info(f'splitting {X_train.shape} {X_test.shape} {y_train.shape} {y_test.shape}')

del X, dependent
gc.collect()

if USE_TRAINED_MODEL:
    ls_reg = joblib.load(f'model_weights/lasso/{file_type}/{target}')
    logger.info('model loaded')
else:
    ls_reg = Lasso(random_state=313)
    logger.info('model built')

    ls_reg.fit(X_train, y_train)
    logger.info('model trained')
    
    joblib.dump(ls_reg, f'/ibex/project/c2205/sara/klebsiella_pneumoniae/model_weights/lasso/{file_type}/{target}')
    logger.info('weights stored')

# Step 6: Make predictions with the best XGBoost model
y_pred_test = ls_reg.predict(X_test)
y_pred_train = ls_reg.predict(X_train)
logger.info('prediction done')

# Step 7: Evaluate the performance
logger.info('**************************************')
logger.info('printing performance metrics for train')
logger.info('**************************************')
metrics_train = RegressionMetrics(y_train, y_pred_train)

logger.info(f'R2: {metrics_train.r2()}')
logger.info(f'R2 score: {metrics_train.r2_score()}')
logger.info(f'NSE: {metrics_train.nse()}')
logger.info(f'RMSE: {metrics_train.rmse()}')
logger.info(f'MSE: {metrics_train.mse()}')
logger.info(f'MAE: {metrics_train.mae()}')

correlation_tr, p_value_tr = spearmanr(y_train, y_pred_train)
logger.info(f"Correlation between predicted and actual values: {correlation_tr}")
logger.info(f"P-value of the correlation: {p_value_tr}")

logger.info('*************************************')
logger.info('printing performance metrics for test')
logger.info('*************************************')

metrics_test = RegressionMetrics(y_test, y_pred_test)

logger.info(f'R2: {metrics_test.r2()}')
logger.info(f'R2 score: {metrics_test.r2_score()}')
logger.info(f'NSE: {metrics_test.nse()}')
logger.info(f'RMSE: {metrics_test.rmse()}')
logger.info(f'MSE: {metrics_test.mse()}')
logger.info(f'MAE: {metrics_test.mae()}')


correlation_test, p_value_test = spearmanr(y_test, y_pred_test)
logger.info(f"Correlation between predicted and actual values: {correlation_test}")
logger.info(f"P-value of the correlation: {p_value_test}")

# Calculate the residual variance (variance of the errors)
residual_variance_train = np.var(y_train - y_pred_train)
residual_variance_test = np.var(y_test - y_pred_test)
# Calculate the total phenotypic variance
total_variance_train = np.var(y_train)
total_variance_test = np.var(y_test)
# Calculate the proportion of variance explained by the model
explained_variance_train = total_variance_train - residual_variance_train
explained_variance_test = total_variance_test - residual_variance_test
# Calculate pseudo-heritability (portion of variance explained by the model)
pseudo_heritability_train = explained_variance_train / total_variance_train if total_variance_train > 0 else np.nan
pseudo_heritability_test = explained_variance_test / total_variance_test if total_variance_test > 0 else np.nan

# Print heritability like other metrics
logger.info(f"Estimated Heritability (h²) - Train Set: {pseudo_heritability_train}")
logger.info(f"Estimated Heritability (h²) - Test Set: {pseudo_heritability_test}")
# If you want to calculate interval coverage, you need to implement quantile predictions for XGBoost separately,
# as it does not natively support quantile regression.
interval_coverage_list.append(np.nan)  # Placeholder as quantile prediction is not included
interval_width_list.append(np.nan)
# Create DataFrames for predictions
test_df = pd.DataFrame({'Test': y_test, 'Prediction': y_pred_test})
train_df = pd.DataFrame({'Train': y_train, 'Prediction': y_pred_train})
# Calculate the range of the dependent variable
range_value = y.max() - y.min()
training_range_value = y_train.max() - y_train.min()
test_range_value = y_test.max() - y_test.min()
range_value_list.append(range_value)
# Save results to CSV
test_df.to_csv(f"data/test_danesh/{file_type}/lasso_test_var_{target}.csv", index=False)
train_df.to_csv(f"data/test_danesh/{file_type}/lasso_train_var_{target}.csv", index=False)
logger.info(f"Range of the 'dependent' variable: {range_value}")
logger.info(f"Range of the 'dependent' variable for training: {training_range_value}")
logger.info(f"Range of the 'dependent' variable for test: {test_range_value}")

# # Save heritability results to CSV
# heritability_df = pd.DataFrame({
#     'Dependent_Variable': target,
#     'Heritability_Train': heritability_train,
#     'Heritability_Test': heritability_test
# })
# heritability_df.to_csv(f"data/test_danesh/heritability/{file_type}_{target}.csv", index=False)

s_string = f"{target},{round(metrics_train.r2(), 4)},{round(metrics_train.r2_score(), 4)},{round(metrics_train.nse(), 4)},{round(metrics_train.rmse(), 4)},{round(metrics_train.mse(), 4)},{round(metrics_train.mae(), 4)},{round(metrics_train.mae()/range_value, 4)},{round(correlation_tr, 4)},{round(p_value_tr, 4)},{round(pseudo_heritability_train, 4)},{round(metrics_test.r2(), 4)},{round(metrics_test.r2_score(), 4)},{round(metrics_test.nse(), 4)},{round(metrics_test.rmse(), 4)},{round(metrics_test.mse(), 4)},{round(metrics_test.mae(), 4)},{round(metrics_test.mae()/range_value, 4)},{round(correlation_test, 4)},{round(p_value_test, 4)},{round(pseudo_heritability_test, 4)}\n"

with open(f"data/test_danesh/{file_type}/lasso/results_{file_type}.txt", 'a') as fp:
    fp.write(s_string)

logger.info('done')