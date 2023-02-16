import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, f_classif, mutual_info_regression
from sklearn_evaluation import plot
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import KFold


def remove_low_variance(X, threshold=0.0):
    """
    Removes columns from X that have lower than the threshold variance

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe that we want to select features from
    threshold : float
        Exclusive threshold that all column's variance should have at least

    Returns
    -------
    X : pd.DataFrame
        Dataframe after excluding columns with no (or little) variance
    """
    
    fs = VarianceThreshold(threshold) # 0.01 = if 99% of data is the same, it removes that column
    fs.fit(X)
    X = X.drop(columns = X.columns[~fs.get_support()])
    return X


def correlation(X, threshold=0.9, method="pearson"):
    """
    Removes correlated features that are above the threshold

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame that we want to select features from
    threshold : float
        Upper limit in absolute value of correlation between variables can have
    
    Returns
    -------
    X : pd.DataFrame
        Dataframe after excluding correlated columns
    """

    corr_matrix = X.corr(method)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(abs(upper[c]) > threshold)]
    X = X.drop(columns=drop)
    return X


def cate_num_cols(X, threshold=0.9):
    """
    Determines which columns are categorical and which are numerical by
    comparing number of unique values with total number of observations

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe that we want to select features from
    threshold : float
        Maximum proportion difference between total number of observations 
        and unique values in a column to consider as a continuous variable

    Returns
    -------
    categorical : array
    numerical : array
        Two arrays with categorical and numerical variables
    """

    is_cate = (X.shape[0] - X.nunique().values) / X.shape[0] > threshold
    categorical, numerical = X.columns[is_cate], X.columns[~is_cate]

    return categorical, numerical


def kbest(X, y, score_func, threshold=0.01):
    """
    Keeps k features from X if a feature scores more than 1% of a max score

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe that we want to select features from
    y : numpy array
    score_func : sklearn.feature_selection univariate filter selection methods
        Determines how to measure features' scores
    threshold : float
        Minimum proportion that a feature's score can be compared to a max feature score

    Returns
    -------
    X : pd.DataFrame
        Dataframe that only keeps the columns with score above the threshold
    """

    fs = SelectKBest(score_func=score_func, k="all")
    fs.fit(X, y)

    result = pd.DataFrame(data={"score": fs.scores_ / max(fs.scores_), "fea": X.columns})
    result = result[result['score'] > threshold]

    X = X[result['fea']]

    return X


def random_forest_selection(X, y, task, imp_threshold=0.85):
    """
    Chooses features based on their importance calculated by random forest.
    Their cutoff is determined by the imp_threshold in a cumulative way.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame that we want to select features from
    y : numpy array
        Target variable
    task : str
        Tells whether task is regression or classification with reg or clf
    imp_threshold : float
        Cutoff for cumulataive random forest feature importance

    Returns
    -------
    fea_lst : list
        List that stores selected features
    """

    if task == "clf":
        rf = RandomForestClassifier()
    else:
        rf = RandomForestRegressor()
    rf.fit(X, y)
    rf_fi = pd.DataFrame(rf.feature_importances_, columns=["score"], 
                         index=rf.feature_names_in_).sort_values(by="score", ascending=False)

    sum_imp = 0
    fea_lst = []
    for index, row in rf_fi.iterrows():
        sum_imp += row[0]
        fea_lst.append(index)
        if sum_imp > imp_threshold:
            break

    return fea_lst


def kfold_mean(X, y, task, eval_metric=None, n_splits=5):
    """
    Performs kfold cross-validation to evaluate which feature set
    gives the best score on average. For regression, it uses 
    mean_squared_error for evaluation. For classification, it uses
    f1_score(weighted)  # TODO is there a better evaluation metric I can implement?

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame that we want to select features from
    y : numpy array
        Target variable
    task : str
        Tells whether task is regression or classification with reg or clf
    n_splits : int
        Number of folds
    eval_metric : ??       # TODO
        Is there a way to implement this?
    
    Returns
    -------
    np.mean(eval_lst) : float
        Mean of cross validation result
    """
    
    kf = KFold(n_splits=n_splits, shuffle=True)

    eval_lst = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train_tmp, y_train_tmp = X.iloc[train_idx], y[train_idx]
        x_test_tmp, y_test_tmp = X.iloc[test_idx], y[test_idx]

        if task == "reg":
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
        model.fit(x_train_tmp, y_train_tmp)
        y_pred = model.predict(x_test_tmp)
        
        if task == "reg":
            eval_lst.append(mean_squared_error(y_test_tmp, y_pred))
        else:
            eval_lst.append(f1_score(y_test_tmp, y_pred, average="weighted"))
    
    return np.mean(eval_lst)


def feature_selection(X, y, task):
    """
    # TODO write function description
    Returns selected features from X using filter methods and random forest
    appropriate to the task stated

    Parameters
    ----------
    X : pd.DataFrame        # TODO can I make datatype more general?
        DataFrame that we want to select features from
    y : numpy array
        Target variable
    task : str
        Tells whether task is regression or classification with reg or clf

    Returns
    # TODO choose what to return
    -------
    
    """
    # TODO Check whether other parameters are correctly given
    if task != "reg" and task != "clf":
        raise ValueError(
            "task needs to be specified by either 'reg' or 'clf'"
        )
    
    X_fs = remove_low_variance(X, 0)
    X_fs_lst = []
    X_fs_lst.append(X_fs)

    # TODO : is it better to determine by myself or ask users?
    categorical, numerical = cate_num_cols(X_fs)

    # peforming filter feature selection
    if len(categorical) == 0:
        if task == "reg":
            X_fs_1 = correlation(X_fs, 0.9, "pearson")
            X_fs_1 = correlation(X_fs_1, 0.9, "spearman")
            if X_fs_1.shape != X_fs_lst[0].shape:
                X_fs_lst.append(X_fs_1)
            
            X_fs_2 = kbest(X_fs_1, y, score_func=mutual_info_regression)
            X_fs_lst.append(X_fs_2)
        else:
            X_fs_1 = kbest(X_fs, y, score_func=f_classif)
            X_fs_2 = kbest(X_fs, y, score_func=mutual_info_classif)

            X_fs_lst.append(X_fs_1)
            X_fs_lst.append(X_fs_2)
    # TODO check with categorical variables dataset / more regulation on what structure of X?
    elif len(numerical) == 0: 
        if task == "reg":
            X_fs_1 = kbest(X_fs, y, score_func=f_classif)

            X_fs_lst.append(X_fs_1)
        else:
            X_fs_1 = kbest(X_fs, y, score_func=mutual_info_classif)
            if min(X_fs.min(axis=1)) >= 0:
                X_fs_2 = kbest(X_fs, y, score_func=chi2) # needs to be non-negative

            X_fs_lst.append(X_fs_1)
            X_fs_lst.append(X_fs_2)
    
    # performing embedded feature selection
    fea_selected_lst = []
    score_lst = []
    for X_i in X_fs_lst:
        X_i_fea = random_forest_selection(X_i, y, task, imp_threshold=0.85)
        fea_selected_lst.append(X_i_fea)
        score_lst.append(kfold_mean(X_i[X_i_fea], y, task))

    
    # using sklearn-evaluation's plot.feature_importances()
    if task == "reg":
        model = RandomForestRegressor()
        model.fit(X[fea_selected_lst[np.argmax(score_lst)]], y)
    else:
        model = RandomForestClassifier()
        model.fit(X[fea_selected_lst[np.argmax(score_lst)]], y)
    # TODO Feature name is lost
    # TODO figure out a better way to use plot.feature_importances
    # TODO figure out a better way to select features
    plot.feature_importances(model)

    return fea_selected_lst, score_lst
