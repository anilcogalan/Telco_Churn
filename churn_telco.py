import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.simplefilter(action="ignore")
pd.set_option('display.max_rows', None, 'display.max_columns', None)

df = pd.read_csv('Telco-Customer-Churn.csv')

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

   It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
   Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold value for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 returned lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################################
# Distribution of numerical and categorical variables in the data
#########################################

def cat_summary(dataframe, col_name, plot=False):
    """
    This function generates a summary of a categorical variable in a pandas DataFrame.
    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the categorical variable
    - col_name (str): the name of the categorical variable to be summarized
    - plot (bool, optional): if True, a countplot will be generated to visualize the distribution of the categorical variable. Defaults to False.
    Returns:
    - None. The function only prints the summary table and, optionally, the countplot.
    Example Usage:
    cat_summary(df, 'color', plot=True)
    This will generate a summary table and a countplot of the 'color' variable in the 'df' DataFrame.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=False)


def num_summary(dataframe, numerical_col, plot=False):
    """
    This function generates a summary of a numerical variable in a pandas DataFrame.
    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the numerical variable
    - numerical_col (str): the name of the numerical variable to be summarized
    - plot (bool, optional): if True, a histogram will be generated to visualize the distribution of the numerical variable. Defaults to False.
    Returns:
    - None. The function only prints the summary table and, optionally, the histogram.
    Example Usage:
    num_summary(df, 'age', plot=True)
    This will generate a summary table and a histogram of the 'age' variable in the 'df' DataFrame.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=False)


#########################################
# Target variable analysis with categorical variables
#########################################

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Generates a summary of the target variable with respect to a specified categorical column in a given pandas DataFrame.

    Args:
    dataframe (pandas.DataFrame): A pandas DataFrame containing the data to be analyzed.
    target (str): The name of the target variable to be analyzed.
    categorical_col (str): The name of the categorical column in the DataFrame to group by.

    Returns:
    None.

    Prints a pandas DataFrame containing the following columns for each unique value in the specified categorical column:
    - TARGET_MEAN: The mean value of the target variable for that category.
    - Count: The number of rows in the DataFrame for that category.
    - Ratio: The percentage of rows in the DataFrame for that category.

    Example usage:
    >>> target_summary_with_cat(df, "SalePrice", "Neighborhood")
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


#########################################
# Outliers
#########################################

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    """
    Calculates the upper and lower limits for outliers in a given column of a pandas DataFrame.

    Args:
    dataframe (pandas.DataFrame): The DataFrame containing the column for which to calculate outlier thresholds.
    col_name (str): The name of the column for which to calculate outlier thresholds.
    q1 (float): The lower percentile threshold. Default is 0.01.
    q3 (float): The upper percentile threshold. Default is 0.99.

    Returns:
    low_limit (float): The lower limit for outliers in the specified column.
    up_limit (float): The upper limit for outliers in the specified column.

    The function calculates the lower and upper limits for outliers in a given column of a pandas DataFrame using the interquartile range (IQR) method. The lower and upper percentile thresholds can be specified with the 'q1' and 'q3' parameters, respectively. The default values for these parameters are 0.01 and 0.99, respectively. The function returns the lower and upper limits for outliers in the specified column as floats.

    Example usage:
    >>> low_limit, up_limit = outlier_thresholds(dataframe, "column_name", q1=0.05, q3=0.95)
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Checks whether there are any outliers in a given column of a pandas DataFrame.

    Args:
    dataframe (pandas.DataFrame): The DataFrame containing the column to check for outliers.
    col_name (str): The name of the column to check for outliers.

    Returns:
    bool: True if there are outliers in the specified column, False otherwise.

    The function uses the 'outlier_thresholds' function to calculate the upper and lower limits for outliers in the specified column, and then checks whether there are any values in the column that are outside of these limits. If there are any such values, the function returns True, indicating the presence of outliers. Otherwise, it returns False.

    Example usage:
    >>> has_outliers = check_outlier(dataframe, "column_name")
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(df, col))
# No Outliers

#########################################
# Missing Value
#########################################

df.isnull().sum()  # [TotalCharges] = 11

#########################################
# Feature Engineering
#########################################

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.head()
df['TotalSpent'] = df['MonthlyCharges'] * df['tenure']

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Specify contract 1 or 2 year customers as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
        x["TechSupport"] != "Yes") else 0, axis=1)

# Young customers with monthly contracts
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
                                       axis=1)

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# People who buy any streaming service
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# average monthly payment
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Current Price increase relative to average price
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# fee per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################################
# Encoding
#########################################

def label_encoder(dataframe, binary_cols):
    le = LabelEncoder()
    dataframe[binary_cols] = le.fit_transform(dataframe[binary_cols])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe


cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]

df = one_hot_encoder(df, cat_cols)
df.head()

#########################################
# Standardization
#########################################

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

#########################################
# Modelling
#########################################

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]


def base_models(X, y):
    """
    Runs several base classification models on the given input features and target variable, and prints the evaluation metrics for each model.

    Args:
    X (pandas.DataFrame or numpy.ndarray): The input features used for prediction.
    y (pandas.Series or numpy.ndarray): The target variable to be predicted.

    Returns:
    None.

    Prints the evaluation metrics for each model:
    - Accuracy
    - AUC (Area Under the ROC Curve)
    - Recall
    - Precision
    - F1-score

    Example usage:
    >>> base_models(X_train, y_train)
    """

    print("Base Models....")
    models = [('LR', LogisticRegression()),
              ("KNN", KNeighborsClassifier()),
              ("CART", DecisionTreeClassifier()),
              ("RF", RandomForestClassifier()),
              ('GBM', GradientBoostingClassifier()),
              ("XGBoost", XGBClassifier(eval_metric='logloss')),
              ("LightGBM", LGBMClassifier()),
              ("CatBoost", CatBoostClassifier(verbose=False))]

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


base_models(X, y)

#########################################
# Hyperparameter Optimization
#########################################

lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

classifiers = [('LR', LogisticRegression(), lr_params),
               ("GBM", GradientBoostingClassifier(), gbm_params),
               ("LightGBM", LGBMClassifier(), lightgbm_params),
               ('CatBoost', CatBoostClassifier(), catboost_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    """
    Performs hyperparameter optimization for several classifiers on the given input features and target variable, and returns the best models with their optimized parameters.

    Args:
    X (pandas.DataFrame or numpy.ndarray): The input features used for prediction.
    y (pandas.Series or numpy.ndarray): The target variable to be predicted.
    cv (int): The number of cross-validation folds to use for model evaluation. Default is 3.
    scoring (str): The evaluation metric to use for model selection. Default is "accuracy".

    Returns:
    best_models (dict): A dictionary containing the best models for each classifier, along with their optimized parameters.

    Prints the evaluation metrics for each classifier before and after hyperparameter optimization, along with the best parameters found during optimization.

    Example usage:
    >>> best_models = hyperparameter_optimization(X_train, y_train, cv=5, scoring="roc_auc")
    """

    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y)

########## LR ##########
# accuracy (Before): 0.8059
# accuracy (After): 0.8068

########## GBM ##########
# accuracy (Before): 0.8005
# accuracy (After): 0.8043

########## LightGBM ##########
# accuracy (Before): 0.7923
# accuracy (After): 0.8022

# CatBoost
# accuracy (After): 0.8035
