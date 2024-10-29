
import sys
import gc
import joblib
import logging
import numpy as np

import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.utils.utils import dateandtime_now

from SeqMetrics import RegressionMetrics

from utils import regression_plot, residual_plot

jobid = sys.argv[1]
target_ID = int(sys.argv[2])
time_of_this_file = dateandtime_now()

SAVE = True
USE_TRAINED_MODEL = False
path_to_model = f'/ibex/user/iftis0a/klebs_ml/model_weights/xgb/XGB_mltF_2_20240821_071817_34917943'

logging.basicConfig(filename=f'log_files/xgb/{dateandtime_now()}_{jobid}_{target_ID}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info(f"job ID: {jobid}")

# Read the input rtab file
df = pd.read_csv(f'data/unitigs_kp.rtab', delimiter='\t', index_col=0)
reference_df = pd.read_csv(f'data/gene_presence_absence.Rtab', delimiter='\t', index_col=0)

logger.info('reading input file')

X = df.T
logger.info(f"transposed X shape: {X.shape}")
logger.info(X.head())

logger.info(f'checking for NaN in X: {X.isna().sum().sum()}')

reference_df = reference_df.T
logger.info(f"transposed reference_df shape: {reference_df.shape}")
logger.info(reference_df.head())

X = X.reindex(reference_df.index)

del df
gc.collect()

logger.info(f"re-indexed X shape: {X.shape}")

logger.info(X.head())

# Read the target rtab file

y = pd.read_csv(f'data/dependent_median_ML.csv')

logger.info(f'checking for NaN in y: {y.isna().sum().sum()}')

logger.info('reading target file')

logger.info(f"y shape: {y.shape}")

logger.info(y.head())

y.index = reference_df.index

logger.info(f"reindexed y shape: {y.shape}")

logger.info(y.head())

del reference_df
gc.collect()

# merging input and target files to ensure same index alingment

input_features = X.columns
output_features = y.columns

logger.info(output_features)

logger.info(f'checking for NaN in X: {X.isna().sum().sum()}')
logger.info(f'checking for NaN in y: {y.isna().sum().sum()}')

data = pd.concat([X, y], axis=1)

logger.info(f'checking for NaN in data: {data.isna().sum().sum()}')

del X, y
gc.collect()

logger.info(f"data shape: {data.shape}")

logger.info(data.head())

target = output_features[target_ID]

logger.info(f'target: {target}')

# splitting data to training and test

logger.info(f'checking for NaN in input and output: {data[input_features].isna().sum().sum()}, {data[target].isna().sum().sum()}')

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(data[input_features], data[target])

logger.info(f'splitting {TrainX.shape} {TestX.shape} {TrainY.shape} {TestY.shape}')

# Combine arrays into a DataFrame
trainX_trainY = pd.DataFrame(TrainX)
trainX_trainY['Y'] = TrainY

trainX_trainY.to_csv(f'data/train_test_split/trainX_trainY_{target}.csv', index=False)

del trainX_trainY
gc.collect()

# Combine arrays into a DataFrame
testX_testY = pd.DataFrame(TestX)
testX_testY['Y'] = TestY

testX_testY.to_csv(f'data/train_test_split/testX_testY_{target}.csv', index=False)

del testX_testY
gc.collect()

# defining model

# model_name = 'XGBRegressor'

# model = Model(
#             model = model_name,
#             input_features=input_features,
#             output_features=target,
#             verbosity=0,
#             )

# model.reset_global_seed(313)

# training

if USE_TRAINED_MODEL:
    model = joblib.load(path_to_model)
    logger.info('model loaded')
else:
    model = XGBRegressor(verbosity=0)
    logger.info('model built')

    model.fit(TrainX, TrainY.values)
    logger.info('model trained')
    
    joblib.dump(model, f'/ibex/project/c2205/sara/klebsiella_pneumoniae/model_weights/xgb/XGB_{target}_{target_ID}_{time_of_this_file}_{jobid}')
    logger.info('weights stored')

# prediction

train_p = model.predict(TrainX)

logger.info('prediction on training data')

test_p = model.predict(TestX)

logger.info('prediction on test data')

train_true_pred = pd.DataFrame({
    'train_true': TrainY,
    'train_pred': train_p,
})

train_true_pred.to_csv('data/train_test_split/train_true_pred.csv', index=False)

test_true_pred = pd.DataFrame({
    'test_true': TestY,
    'test_pred': test_p,
})

test_true_pred.to_csv('data/train_test_split/test_true_pred.csv', index=False)

# evaluation

logger.info('printing performance metrics for training')

metrics_train = RegressionMetrics(TrainY, train_p)

logger.info(f'R2: {metrics_train.r2()}')
print(f'R2: {metrics_train.r2()}')

logger.info(f'R2 score: {metrics_train.r2_score()}')
print(f'R2 score: {metrics_train.r2_score()}')

logger.info(f'NSE: {metrics_train.nse()}')
print(f'NSE: {metrics_train.nse()}')

logger.info(f'RMSE: {metrics_train.rmse()}')
print(f'RMSE: {metrics_train.rmse()}')

logger.info(f'MSE: {metrics_train.mse()}')
print(f'MSE: {metrics_train.mse()}')

logger.info(f'MAE: {metrics_train.mae()}')
print(f'MAE: {metrics_train.mae()}')

# %%

logger.info('printing performance metrics for test')

metrics_test = RegressionMetrics(TestY, test_p)

logger.info(f'R2: {metrics_test.r2()}')
print(f'R2: {metrics_test.r2()}')

logger.info(f'R2 score: {metrics_test.r2_score()}')
print(f'R2 score: {metrics_test.r2_score()}')

logger.info(f'NSE: {metrics_test.nse()}')
print(f'NSE: {metrics_test.nse()}')

logger.info(f'RMSE: {metrics_test.rmse()}')
print(f'RMSE: {metrics_test.rmse()}')

logger.info(f'MSE: {metrics_test.mse()}')
print(f'MSE: {metrics_test.mse()}')

logger.info(f'MAE: {metrics_test.mae()}')
print(f'MAE: {metrics_test.mae()}')

# %%

# visualization

regression_plot(TestY, test_p, f'xgb_rgr_reg_{jobid}_{target}')

# %%

residual_plot(
    train_true=TrainY.values,
    train_prediction=train_p,
    test_true=TestY.values,
    test_prediction=test_p
)
if SAVE:
    plt.savefig(f"figures/xgb_rgr_residual_{jobid}_{target}.png", dpi=600, bbox_inches="tight")