from general_model_building import model_training_evaluation as mte
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
# import xgboost as xgb
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import time
from joblib import dump
import warnings

# import statsmodels.api as sm


start_time = time.time()

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


# adding labels
data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'


demo = 'race' # 'religion1' # 'religion2' # 'gender' # 'race' #'orientation'  #
df_features = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_features.csv')
df_labels = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_diff.csv')
print(df_labels.head())
df_labels = df_labels['diff_perplex']

data = df_features.join(df_labels)
print(data.shape)

data = data[(data['diff_perplex'] < 300) & (data['diff_perplex'] > -300)]
print('After filtering diff perplexity {}'.format(data.shape))

X = data.drop('diff_perplex', axis=1)
y = data['diff_perplex']

train_features, test_features, train_target, test_target = train_test_split(X, y, test_size=0.33, random_state=42)


print('Usable Features of train set')
print(train_features.columns)
print(train_features.head())

'''
cross_val = StratifiedKFold(n_splits=10, random_state=42)

# Uncomment any one classifier to use it for training
# clf = LogisticRegression()
# params = [{'solver': ['newton-cg', 'lbfgs'], 'max_iter': [100, 500, 1000], 'penalty':['l2'], 'C':[10]}]
# params = [{'solver': ['liblinear', 'saga'], 'max_iter': [500, 1000], 'penalty':['l1'], 'C':[0.001, 0.1, 1, 10]}]
#
# clf = SVC()
# params = [{'decision_function_shape': ['ovo'], 'C': [0.01, 0.5, 1], 'kernel': ['rbf', 'linear'], 'probability': [True]}]
#
# clf = RandomForestClassifier()
# params = [{'n_estimators': [1000], 'max_depth': [10], 'min_samples_split': [3], 'max_features': [4]}]
#
# clf = KNeighborsClassifier()
# params = [{'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'n_neighbors': [5, 7, 10], 'metric':['euclidean', 'manhattan'], 'weights':['uniform', 'distance']}]
#
# clf = SGDClassifier()
# params = [{'epsilon': [0.1, 0.5], 'loss': ['log','modified_huber'], 'penalty': ['l1', 'l2']}]

# Running a gridsearch for a given model and parameterset
clf = GradientBoostingClassifier()
params = [{'learning_rate': [0.1], 'loss': ['exponential'], 'n_estimators': [75], 'subsample':[0.5], 'max_depth':[10]}]

print('\n Assessing for all data:')
best_clf, best_comb, best_y_pred, best_y_proba_pred = \
    mte.some_gs_Funk(clf, train_features, train_target, cross_val, params, verbose=False)

print('Feature importance for classifier - ')
try:
    for column, feature_importance in zip(train_features.columns, best_clf.feature_importances_):
        print(column, feature_importance)
except Exception as e:
    print(Exception.__name__)

dump(best_clf, '../models/classifier_GB.joblib')

# Analysing Logistic regression as base classifier to check significance of generated features
logit_model=sm.Logit(np.array(train_target, dtype=float),np.array(train_features, dtype=float))
result=logit_model.fit()
print('Logit statsmodel')
print(result.summary())

# Uncomment below code to train meta classifier

# Running a grid search for various models and parametersets in order to use resulting probabilities as data for meta classifier
clf_list = [GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier()]
params_list = [[{'learning_rate': [0.1], 'loss': ['exponential'], 'n_estimators':[100], 'subsample':[0.8], 'max_depth':[10]}],
               [{'n_estimators': [100], 'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_split': [3]}],
               [{'learning_rate': [0.5], 'n_estimators': [500], 'random_state': [0]}]]

data_list = [train_features, train_features, train_features]
meta_data_clf = pd.DataFrame()
i = 1
print('\n Assessing level 1 models:')
for clf, params, data in zip(clf_list, params_list, data_list):
    print('\n Running classifier: %i' %i)

    best_clf, best_comb, best_y_pred, best_y_proba_pred = \
        mte.some_gs_Funk(clf, data, train_target, cross_val, params, verbose=False,
                             profit_relevant=False)

    print('Feature importance for classifier %i - ' %i)

    try:
        for column, feature_importance in zip(data.columns, best_clf.feature_importances_):
            print(column, feature_importance)
    except Exception as e:
        print(Exception.__name__)

    meta_data_clf['Classifier %i' %i] = best_y_proba_pred
    i += 1

# Building ensemble model
for column in meta_data_clf:
    mean = meta_data_clf[column].mean()
    sd = meta_data_clf[column].std()
    meta_data_clf[column] = (meta_data_clf[column] - mean) / sd

# Combine data for stacking including the original features (if to be used)
comb_data = pd.concat([meta_data_clf, train_features], axis=1, ignore_index=True)

# Choose/ initialise meta classifier and its parameters for grid search
# meta_clf = LogisticRegression()
# meta_params = [{'solver': ['newton-cg'], 'max_iter': [1000], 'multi_class': ['multinomial']}]

# meta_clf = RandomForestClassifier()
# meta_params = [{'n_estimators': [100, 1000], 'max_depth': [5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [3, 5]}]

meta_clf = GradientBoostingClassifier()
meta_params = [{'learning_rate': [0.003, 0.01, 0.1], 'loss': ['exponential', 'deviance'], 'n_estimators':[75, 100], 'subsample':[0.5, 0.8]}]

print('\n Assessing meta classifier:')
best_clf, best_comb, best_y_pred, best_y_proba_pred = mte.some_gs_Funk(meta_clf, meta_data_clf,
                                                                                  train_target, cross_val, meta_params,
                                                                                  verbose=False, profit_relevant=False)


dump(best_clf, '../models/classifier_ensemble.joblib')

print('\n------------End of Classification-----------\n')
'''
train_features_reg = train_features
train_target_reg = train_target

print(train_features_reg.isna().sum())
print('Training samples for Level 2:'.format(train_features_reg.shape[0]))

# Get features and target variable for training regression model
# train_features_reg, train_target_reg = train_reg.reindex(columns=['touristic_focus', 'text_length', 'trusted_tld', 'adj_percent', 'suspicious_words_count',
#                                                       'start_subDl_duration_days', 'neg_duration', 'committee_number', 'total_committee_citation',
#                                                       'total_committee_publications', 'total_committee_avg_citation', 'avg_cit_per_person',
#                                                       'conf_series_citation', 'private_registration', 'identity_hidden', 'completeness',
#                                                       'geo_na_eu', 'geo_asia', 'whois_na_eu', 'whois_asia', 'geo_loc_diff_whois_loc',
#                                                       'website_age']), train_reg['fcr']


print('Features of train set')
print(train_features_reg.columns)
print(train_features_reg.head())

# Uncomment below code to normalise train data
'''
train_features_reg['text_length'] = (train_features_reg['text_length']-train_features_reg['text_length'].mean())/train_features_reg['text_length'].std()
train_features_reg['start_subDl_duration_days'] = (train_features_reg['start_subDl_duration_days']-train_features_reg['start_subDl_duration_days'].mean())/train_features_reg['start_subDl_duration_days'].std()
train_features_reg['suspicious_words_count'] = (train_features_reg['suspicious_words_count']-train_features_reg['suspicious_words_count'].mean())/train_features_reg['suspicious_words_count'].std()
train_features_reg['total_committee_citation'] = (train_features_reg['total_committee_citation']-train_features_reg['total_committee_citation'].mean())/train_features_reg['total_committee_citation'].std()
train_features_reg['total_committee_publications'] = (train_features_reg['total_committee_publications']-train_features_reg['total_committee_publications'].mean())/train_features_reg['total_committee_publications'].std()
train_features_reg['total_committee_avg_citation'] = (train_features_reg['total_committee_avg_citation']-train_features_reg['total_committee_avg_citation'].mean())/train_features_reg['total_committee_avg_citation'].std()
train_features_reg['avg_cit_per_person'] = (train_features_reg['avg_cit_per_person']-train_features_reg['avg_cit_per_person'].mean())/train_features_reg['avg_cit_per_person'].std()
train_features_reg['conf_series_citation'] = (train_features_reg['conf_series_citation']-train_features_reg['conf_series_citation'].mean())/train_features_reg['conf_series_citation'].std()
train_features_reg['website_age'] = (train_features_reg['website_age']-train['website_age'].mean())/train_features_reg['website_age'].std()
'''
cross_val_reg = KFold(n_splits=10, random_state=42)

# Uncomment any one classifier to train regession model
# reg = LinearRegression()
# params_reg = [{'fit_intercept': [True], 'normalize': [True]}]
#
# reg = Ridge()
# params_reg = [{'alpha': [0.1, 1, 2], 'fit_intercept': [True, False], 'solver':['auto', 'svd', 'sag']}]
#
# reg = DecisionTreeRegressor()
# params_reg = [{'criterion': ['mse', 'friedman_mse'], 'max_depth': [None, 5, 7, 10], 'max_features':[None, 'auto', 'sqrt', 'log2'],
#                'max_leaf_nodes':[None, 50, 100, 150], 'min_samples_leaf':[1, 2, 3]}]
#
# reg = xgb.XGBRegressor()
# params_reg = [{'colsample_bytree': [0.3, 0.6, 0.8, 1], 'gamma': [1], 'learning_rate':[0.009, 0.01, 0.03], 'max_depth':[3],
#                'n_estimators':[50, 100, 250], 'subsample':[0.8]}]
#
# reg = KernelRidge()
# params_reg = [{'alpha': [1, 2], 'kernel': ['linear', 'sigmoid'], 'gamma':[0.1, 3.0, 1], 'degree':[3, 4 ,5]}]

reg = lgb.LGBMRegressor()
params_reg = [{'num_leaves': [10, 30, 50], 'learning_rate': [0.05, 0.01, 0.1], 'n_estimators': [20, 50, 100]}]

# reg = RandomForestRegressor()
# params_reg = [{'n_estimators': [1000], 'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_split': [3]}]

# reg = LogisticRegression()
# params_reg = [{'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['liblinear', 'lbfgs', 'saga']}]

print('\n Assessing for all data:')
best_reg, best_comb_reg, best_y_pred_reg = \
    mte.some_gs_Funk_reg(reg, train_features_reg, train_target_reg, cross_val_reg, params_reg, verbose=False, r2_score_relevant=False)

try:
    for column, feature_importance in zip(train_features_reg.columns, best_reg.feature_importances_):
        print(column, feature_importance)
except Exception as e:
    print(Exception.__name__)

# dump(best_reg, '../models/regressor_RF.joblib')

# Uncomment below code to train meta regression model

# Running a grid search for various models and parametersets in order to use resulting probabilities as data for meta classifier
# reg_list = [GradientBoostingRegressor(), xgb.XGBRegressor(), RandomForestRegressor()]
# params_reg_list = [[{'loss': ['huber'], 'learning_rate': [0.03], 'n_estimators':[1000], 'subsample':[0.8]}],
#                    [{'gamma': [1], 'learning_rate': [0.1], 'max_depth': [3], 'n_estimators':[100], 'subsample':[0.8]}],
#                    [{'n_estimators': [1000], 'max_depth': [10], 'max_features': ['sqrt'],
#                      'min_samples_split': [3]}]]
# data_reg_list = [train_features_reg, train_features_reg, train_features_reg]
# meta_data_reg = pd.DataFrame()
# i = 1
# print('\n Assessing level 1 models:')
# for reg, params_reg, data_reg in zip(reg_list, params_reg_list, data_reg_list):
#     print('\n Running regressor: %i' %i)
#
#     best_reg, best_comb_reg, best_y_pred_reg = \
#         mte.some_gs_Funk_reg(reg, data_reg, train_target_reg, cross_val_reg, params_reg, verbose=False,
#                              r2_score_relevant=False)
#
#     print('Feature importance for regressor %i - ' %i)
#
#     try:
#         for column, feature_importance in zip(data_reg.columns, best_reg.feature_importances_):
#             print(column, feature_importance)
#     except Exception as e:
#         print(Exception.__name__)
#
#     meta_data_reg['Regressor %i' %i] = best_y_pred_reg
#     i += 1
#
# for column in meta_data_reg:
#     mean = meta_data_reg[column].mean()
#     sd = meta_data_reg[column].std()
#     meta_data_reg[column] = (meta_data_reg[column] - mean) / sd
#
# # Combine data for stacking including the original features (if to be used)
# comb_data = pd.concat([meta_data_reg, train_target_reg], axis=1, ignore_index=True)
#
# # Choose/ initialise meta classifier and its parameters for grid search
# # meta_reg = xgb.XGBRegressor()
# # meta_params_reg = [{'gamma': [1], 'learning_rate': [0.03], 'max_depth':[3],
# #                     'n_estimators':[250], 'subsample':[0.8]}]
#
# meta_reg = RandomForestRegressor()
# meta_params_reg = [{'n_estimators': [100, 1000], 'max_depth': [5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [3, 5]}]
#
# print('\n Assessing meta regressor:')
# best_reg, best_comb_reg, best_y_pred_reg = mte.some_gs_Funk_reg(meta_reg, meta_data_reg, train_target_reg, cross_val_reg, meta_params_reg, verbose=False)
#
# dump(best_reg, '../models/regressor_ensemble.joblib')


print('\n------------End of Regression-----------\n')

print("Code executed in - {}".format(time.time() - start_time))