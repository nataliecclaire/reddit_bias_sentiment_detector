from general_model_building import evaluation
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
from math import sqrt


def someFunk(clf, X, y, cross_val, verbose=True, early_stopping=None, smote=None, profit_relevant=False):
    """
    takes a classifier, data and labels, evaluates performance of classifier using cross validation
    and returns trained model as well as (crossvalidation based) predictions
    Parameters
    ----------
    clf : classifier
        The parameterised classifier to be used (requires sklearn interface)
    X : pd.DataFrame
        DataFrame with all the features used in model
    y : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with true labels
    cross_val : cross validation object
        sklearn cross validation object to specify folds etc.
    verbose : bool
        Set to false if less results should be printed
    early_stopping : bool
        Parameter for regularisation used for LGBM classifier
    smote : smote object
        In case over or undersampling should be used a smote object can be passed
    profit_relevant : bool
        Specify if profit function should be used as main evaluation criterion instead of AUC

    Returns
    -------
    trained classifier + labels and probabilities coming from cross validation
    """

    i = 0
    # initialise arrays to be filled in each split
    y_all = np.array([])
    y_pred_all = np.array([])
    y_pred_proba_all = np.array([])
    index_all = np.array([])
    # Model Training
    for (train_index, val_index) in cross_val.split(X, y):
        # cross-validation randomly splits train data into train and validation data
        if verbose:
            print('\n Fold %d' % (i + 1))

        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        x_train, x_val = np.array(x_train), np.array(x_val)
        y_train, y_val = np.array(y_train), np.array(y_val)

        if smote is not None:
            # if some object is given, over/undersample accordingly before prediction
            x_train_res, y_train_res = smote.fit_sample(x_train, y_train)
            if early_stopping is None:
                clf.fit(x_train_res, y_train_res)
            else:
                # if early stopping is turned on, use it in fitting of classifier
                clf.fit(x_train_res, y_train_res, early_stopping_rounds=early_stopping,
                        # eval_metric='auc',
                        eval_set=[(x_val, y_val)], verbose=verbose)
        else:
            if early_stopping is None:
                clf.fit(x_train, y_train)
            else:
                # if early stopping is turned on, use it in fitting of classifier
                clf.fit(x_train, y_train, early_stopping_rounds=early_stopping,
                        # eval_metric='auc',
                        eval_set=[(x_val, y_val)], verbose=verbose)

        # predict validation set and get eval metrics
        y_pred_val = clf.predict(x_val)
        y_pred_proba_val = clf.predict_proba(x_val)[:, 1]

        eval(y_val, y_pred_val, y_pred_proba_val, 'eval', verbose)

        y_all, y_pred_all, y_pred_proba_all = np.append(y_all, y_val), np.append(y_pred_all, y_pred_val), np.append(y_pred_proba_all, y_pred_proba_val)
        index_all = np.append(index_all, val_index)
        i = i + 1
    if verbose:
        print('\n Overall results:')
        eval(y_all, y_pred_all, y_pred_proba_all, 'overall', verbose, profit_relevant=profit_relevant)

    if verbose:
        print(evaluation.confusion_matrix_report(y_all, y_pred_all))

    # sort predictions to fit original order (which was destroyed by cross validation)
    meta_data = pd.DataFrame()
    meta_data['pred'] = y_pred_all
    meta_data['proba'] = y_pred_proba_all
    meta_data['index'] = index_all
    meta_data = meta_data.sort_values(['index'])
    y_pred_all = np.array(meta_data['pred'])
    y_pred_proba_all = np.array(meta_data['proba'])

    # return model (fitted to full train data) for evaluation and prediction
    X, y = np.array(X), np.array(y)
    clf.fit(X, y)

    return clf, y_pred_all, y_pred_proba_all


def some_gs_Funk(clf, X, y, cross_val, params, verbose=True, early_stopping=None, smote=None, profit_relevant=False):
    """
    Can be used to run someFunk using gridsearch
    Parameters
    ----------
    clf : classifier
        The parameterised classifier to be used (requires sklearn interface)
    X : pd.DataFrame
        DataFrame with all the features used in model
    y : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with labels
    cross_val : cross validation object
        sklearn cross validation object to specify folds etc.
    params : dict
        dictionary specifying all combinations of parameters to be tested
    verbose : bool
        Set to false if less results should be printed
    early_stopping : bool
        Parameter for regularisation used for LGBM classifier
    smote : smote object
        In case over or undersampling should be used a smote object can be passed
    profit_relevant : bool
        Specify if profit function should be used as main evaluation criterion instead of AUC

    Returns
    -------
    Best classifier, best parameter combination as well as its predicted labels and confidence scores
    """

    grid = ParameterGrid(params)
    # Initialise variables to save best values
    best_profit = -100000
    best_auc = -100000
    best_clf = None
    best_comb = None
    best_y_pred = None
    # iterate over all specified parameter combinations
    for comb in grid:
        clf, y_pred_all, y_pred_proba_all = someFunk(clf.set_params(**comb), X, y, cross_val,
                                                     early_stopping=early_stopping, verbose=verbose, smote=smote)
        print(comb)
        if profit_relevant:
            profit = evaluation.cost_scorer_2(y, y_pred_all)
        auc = roc_auc_score(y, y_pred_proba_all)

        # If current model is better than currently best model fill corresponding variables
        if profit_relevant:
            if profit > best_profit:
                best_profit = profit
                best_auc = auc
                best_clf = clf
                best_comb = comb
                best_y_pred = y_pred_all
                best_y_proba_pred = y_pred_proba_all
        else:
            if auc > best_auc:
                best_auc = auc
                best_clf = clf
                best_comb = comb
                best_y_pred = y_pred_all
                best_y_proba_pred = y_pred_proba_all

        if verbose:
            eval(y, y_pred_all, y_pred_proba_all, '', profit_relevant=profit_relevant)
        else:
            if profit_relevant:
                print('-Profit: %.6f' %profit)
            print('-AUC: %.6f' % auc)

    # Print information on and evaluation of best model
    if profit_relevant:
        print('\n\nbest profit is {} with params {} '.format(best_profit, best_comb))
    else:
        print('\n\nbest AUC is {} with params {} '.format(best_auc, best_comb))
    print('\nbreakdown:')
    eval(y, best_y_pred, best_y_proba_pred, 'best result')
    print('\nconfusion matrix:')
    print(evaluation.confusion_matrix_report(y, best_y_pred))


    return best_clf, best_comb, best_y_pred, best_y_proba_pred


def eval(y, y_pred, y_pred_proba, name, verbose=True, profit_relevant=False):
    """
    Evaluates predictions based on metrics we focus on
    Parameters
    ----------
    y : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with true labels
    y_pred : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with predicted labels
    y_pred_proba : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with confidence scores
    name : str
        name to be printed as part of evaluation
    verbose : bool
        Set to False if less results should be printed
    profit_relevant : bool
        Specify if profit function should be used as main evaluation criterion instead of AUC

    Returns
    -------
    Nothing, only for printing
    """

    if np.sum(y_pred) != 0:
        pp = precision_score(y, y_pred)
    else:
        pp = 0
    re = recall_score(y, y_pred)
    if profit_relevant:
        profit = evaluation.cost_scorer_2(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    if verbose:
        print(name + '-Precision: %.6f' % pp)
        print(name + '-Recall: %.6f' % re)
        if profit_relevant:
            print(name + '-Profit: %.6f' % profit)
        print(name + '-AUC: %.6f' % auc)

    # return pp, re, profit


def someFunk_reg(clf, X, y, cross_val, verbose=True, early_stopping=None, smote=None, r2_score_relevant=False):
    """
       takes a regressor, data and labels, evaluates performance of classifier using cross validation
       and returns trained model
       Parameters
       ----------
       clf : regressor
           The parameterised regressor to be used (requires sklearn interface)
       X : pd.DataFrame
           DataFrame with all the features used in model
       y : pd.DataFrame or pd.Series or np.array
           One column DataFrame/ array or Series with true labels
       cross_val : cross validation object
           sklearn cross validation object to specify folds etc.
       verbose : bool
           Set to false if less results should be printed
       early_stopping : bool
            Parameter for regularisation used for LGBM classifier
       smote : smote object
           In case over or undersampling should be used a smote object can be passed
       r2_score_relevant : bool
           Specify if R2 score should be used as main evaluation criterion instead of RMSE

       Returns
       -------
       trained classifier + labels coming from cross validation
       """
    i = 0
    # initialise arrays to be filled in each split
    y_all = np.array([])
    y_pred_all = np.array([])
    index_all = np.array([])
    # Model Training
    for (train_index, val_index) in cross_val.split(X, y):
        # cross-validation randomly splits train data into train and validation data
        if verbose:
            print('\n Fold %d' % (i + 1))

        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        x_train, x_val = np.array(x_train), np.array(x_val)
        y_train, y_val = np.array(y_train), np.array(y_val)

        if smote != None:
            x_train_res, y_train_res = smote.fit_sample(x_train, y_train)
            if early_stopping == None:
                clf.fit(x_train_res, y_train_res)
            else:
                clf.fit(x_train_res, y_train_res, early_stopping_rounds=early_stopping,
                        # eval_metric='auc',
                        eval_set=[(x_val, y_val)], verbose=verbose)
        else:
            if early_stopping == None:
                clf.fit(x_train, y_train)
            else:
                clf.fit(x_train, y_train, early_stopping_rounds=early_stopping,
                        # eval_metric='auc',
                        eval_set=[(x_val, y_val)], verbose=verbose)


        # predict validation set and get eval metrics
        y_pred_val = clf.predict(x_val)
        # print(y_pred_val)

        eval_reg(y_val, y_pred_val, 'eval', verbose, r2_score_relevant=r2_score_relevant)

        y_all, y_pred_all = np.append(y_all, y_val), np.append(y_pred_all, y_pred_val)
        index_all = np.append(index_all, val_index)
        i = i + 1
    if verbose:
        print('\n Overall results:')
        eval_reg(y_all, y_pred_all, 'overall', verbose, r2_score_relevant=r2_score_relevant)

    # return model (fitted to full train data) for evaluation and prediction
    X, y = np.array(X), np.array(y)
    clf.fit(X, y)

    meta_data = pd.DataFrame()
    meta_data['pred'] = y_pred_all
    meta_data['index'] = index_all
    meta_data = meta_data.sort_values(['index'])
    y_pred_all = np.array(meta_data['pred'])

    return clf, y_pred_all


def some_gs_Funk_reg(clf, X, y, cross_val, params, verbose=True, early_stopping=None, smote=None, r2_score_relevant=False):
    """
       Can be used to run someFunk using gridsearch
       Parameters
       ----------
       clf : classifier
           The parameterised classifier to be used (requires sklearn interface)
       X : pd.DataFrame
           DataFrame with all the features used in model
       y : pd.DataFrame or pd.Series or np.array
           One column DataFrame/ array or Series with labels
       cross_val : cross validation object
           sklearn cross validation object to specify folds etc.
       params : dict
           dictionary specifying all combinations of parameters to be tested
       verbose : bool
           Set to false if less results should be printed
       early_stopping : bool
           Parameter for regularisation used for LGBM classifier
       smote : smote object
           In case over or undersampling should be used a smote object can be passed
       r2_score_relevant : bool
           Specify if R2 score should be used as main evaluation criterion instead of RMSE

       Returns
       -------
       Best regressor, best parameter combination as well as its predicted labels
       """
    grid = ParameterGrid(params)
    # Initialise variables to save best values
    best_rmse = 10000
    best_r2 = -10
    best_clf = None
    best_comb = None
    best_y_pred = None
    for comb in grid:
        clf, y_pred_all = someFunk_reg(clf.set_params(**comb), X, y, cross_val, early_stopping=early_stopping,
                                       verbose=verbose, smote=smote, r2_score_relevant=r2_score_relevant)

        # print(sum(np.isnan(y_pred_all)))
        #ToDo ggf. anpassen mit **comb für xbg, lgbt
        print(comb)
        if r2_score_relevant:
            r_2_score = r2_score(y, y_pred_all)
        else:
            rmse = sqrt(mean_squared_error(y, y_pred_all))

        # If current model is better than currently best model fill corresponding variables
        if r2_score_relevant:
            if r_2_score > best_r2:
                best_r2 = r_2_score
                best_clf = clf
                best_comb = comb
                best_y_pred = y_pred_all
                # best_y_proba_pred = y_pred_proba_all
        else:
            if rmse < best_rmse:
                best_rmse = rmse
                best_clf = clf
                best_comb = comb
                best_y_pred = y_pred_all
                # best_y_proba_pred = y_pred_proba_all


        if verbose:
            eval_reg(y, y_pred_all, '', r2_score_relevant=r2_score_relevant)
        else:
            if r2_score_relevant:
                print('-R2 score: %.6f' % r_2_score)
            else:
                print('-RMSE: %.6f' % rmse)

    # Print information on and evaluation of best model
    if r2_score_relevant:
        print('\n\nbest R2 score is {} with params {} '.format(best_r2, best_comb))
    else:
        print('\n\nbest RMSE is {} with params {} '.format(best_rmse, best_comb))
    print('\nbreakdown:')
    eval_reg(y, best_y_pred, 'best result', r2_score_relevant=r2_score_relevant)

    return best_clf, best_comb, best_y_pred


def eval_reg(y, y_pred, name, verbose=True, r2_score_relevant=False):
    """
    Evaluates predictions based on metrics we focus on
    Parameters
    ----------
    y : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with true labels
    y_pred : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with predicted labels
    name : str
        name to be printed as part of evaluation
    verbose : bool
        Set to False if less results should be printed
    r2_score_relevant : bool
        Specify if R2 score should be used as main evaluation criterion instead of RMSE

    Returns
    -------
    Nothing, only for printing
    """
    # Evaluates predictions based on metrics we focus on
    if r2_score_relevant:
        r2 = r2_score(y, y_pred)
    else:
        rmse = sqrt(mean_squared_error(y, y_pred))
    if verbose:
        if r2_score_relevant:
            print(name + '-R2 score: %.6f' % r2)
        else:
            print(name + '-RMSE: %.6f' % rmse)