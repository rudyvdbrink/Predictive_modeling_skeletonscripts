#%% import dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


#%% functions for preprocessing

#check for duplicates
def check_duplicates(df):
    """Checks and removes duplicate rows from a dataframe.

    Args:
        df (pandas.DataFrame): Raw DataFrame

    Returns:
        df (pandas.DataFrame): DataFrame with duplicate rows removed.
    """    
    has_dup = df.duplicated()
    true_dup = np.where(has_dup == True)
    if len(true_dup[0]) > 0:
        print("Data has", len(true_dup[0]), "duplicates")
        df.drop_duplicates(keep='first', inplace=True)
    else:
        print("No duplicates found")
    return df

#function for one-hot encoding
def one_hot(df, column_names):
    """Converts columns in a dataframe to one-hot encoded variants. The original columns are removed. The first one-hot encoded column is dropped.

    Args:
        df (pandas.DataFrame): Raw DataFrame
        column_names (list): list with the names of the columns to one-hot encode

    Returns:
        df (pandas.DataFrame): DataFrame with one-hot encoded columns.
    """    
    for col in column_names:
        dummies = pd.get_dummies(df[[col]].astype('category'),drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([col], axis=1)
    return df

#function for adding interaction terms between all columns
def add_full_interactions(df,degree=2):
    """Add interaction terms between all columns in a dataframe. 

    Args:
        df (pandas.DataFrame): Raw DataFrame
        degree (int, optional): Degree of polynomial to compute. Defaults to 2.

    Returns:
        df (pandas.DataFrame): DataFrame with interaction terms.
    """    
    poly = PolynomialFeatures(interaction_only=True, include_bias=False, degree=degree)
    X_poly = poly.fit_transform(df)
    column_names = list(poly.get_feature_names_out())
    X_poly = pd.DataFrame(X_poly,columns=column_names)
    return X_poly

#function for adding interaction terms between select columns
def add_interactions(df,columns,degree=2):
    """Add interaction terms between specific columns in a dataframe. 

    Args:
        df (pandas.DataFrame): Raw DataFrame
        columns (list): list of column names between which interactions are computed.
        degree (int, optional): _description_. Defaults to 2.

    Returns:
        df (pandas.DataFrame): DataFrame with interaction terms.
    """    
    poly = PolynomialFeatures(interaction_only=True, include_bias=False, degree=degree)
    X_poly = poly.fit_transform(df[columns])
    column_names = list(poly.get_feature_names_out())
    X_poly = pd.DataFrame(X_poly,columns=column_names)
    X_poly = pd.concat([df, X_poly.drop(columns,axis=1)], axis=1)
    return X_poly

#function for computing sigmoid (e.g. to look at logistic regression fit)
def sigmoid(x,b):
    """Compute a sigmoid function.

    Args:
        x (int, float): x-axis values
        b (int, float): bias term

    Returns:
        signmoid (numpy.array): the sigmoid function
    """    
    return 1 / (1 + np.exp(-(b+x)))

#min-max scale vector to pre-specified range
def linmap(vector, new_min, new_max):
    """Linearly map a vector onto a new range.

    Args:
        vector (np.array): vector of numbers
        new_min (int, float): new minimum for the vector 
        new_max (int, float): new maximum for the vector 

    Returns:
        scaled_vector: vector mapped onto the new range of values
    """    
    vector = np.array(vector)
    old_min = np.min(vector)
    old_max = np.max(vector)
    
    # Avoid division by zero if the old_min equals old_max
    if old_min == old_max:
        return np.full_like(vector, new_min if old_min == old_max else new_max)
    
    # Scale the vector
    scaled_vector = (vector - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_vector

def filter_features(X_train,X_test,thresh=0.95):
    """Filters features in a design matrix by spearman correlation coefficient. Features that correlate above a threshold are removed, column-wise. 

    Features are identified based on the training set. The same features are removed from the testing set.

    Args:
        X_train (pandas.DataFrame): Design matrix for the training set.
        X_test (pandas.DataFrame): Design matrix for the testing set.
        thresh (float, optional): Spearman's rho to use as threshold. Defaults to 0.95.

    Returns:
        X_train (pandas.DataFrame): Design matrix for the training set.
        X_test (pandas.DataFrame): Design matrix for the testing set.
    """    
    
    #compute correlations and get half the correlations
    cm = X_train.corr(method = "spearman").abs() #compute correlation matrix    
    upper = cm.where(np.triu(np.ones(cm.shape), k = 1).astype(bool)) #select upper triangle of matrix

    #find index / indices of feature(s) with correlation above threshold
    columns_to_drop = [column for column in upper.columns if any(upper[column] > thresh)]

    # Drop features
    X_train = X_train.drop(columns_to_drop, axis = 1)
    X_test = X_test.drop(columns_to_drop, axis = 1)

    # Print the number of features to be dropped
    num_features_dropped = len(columns_to_drop)
    if num_features_dropped > 0:
        print(f"Dropping {num_features_dropped} features due to high correlation.")
    else:
        print("No features dropped based on correlation.")

    return X_train, X_test, columns_to_drop

def undo_log_transform(data):
    """
    Undo a log transform on a pandas Series or numpy array.
    
    Parameters:
    data (pd.Series or np.ndarray): The data that has been log-transformed.
    
    Returns:
    pd.Series or np.ndarray: The data after undoing the log transformation.
    """
    
    # Define the transformation function
    def transform(x):
        if x <= 0:
            return 0
        else:
            return np.exp(x)
    
    # If input is a pandas Series
    if isinstance(data, pd.Series):
        return data.apply(transform)
    
    # If input is a numpy array
    elif isinstance(data, np.ndarray):
        vectorized_transform = np.vectorize(transform)
        return vectorized_transform(data)
    
    else:
        raise TypeError("Input should be a pandas Series or numpy array")
    
def prediction_plots(y_train, y_test, y_pred_train, y_pred):
    """
    Evaluate a model by plotting scatter plots of true vs predicted values
    with a least squares regression line and MSE. 
    
    Parameters:
    y_train (array-like): True values for the training set.
    y_test (array-like): True values for the test set.
    y_pred_train (array-like): Predicted values for the training set.
    y_pred (array-like): Predicted values for the test set.
    """
    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test  = mean_squared_error(y_test,  y_pred)

    # Scatter plot for the training set
    axs[0].scatter(y_train, y_pred_train, alpha=0.5, label='Data')
    
    # Regression line for training set
    m_train, b_train = np.polyfit(y_train, y_pred_train, 1)
    axs[0].plot(y_train, m_train * y_train + b_train, color='blue', label=f'Fit: y={m_train:.2f}x + {b_train:.2f}')

    # Perfect prediction line for reference
    axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Prediction')

    axs[0].set_xlabel('True Values')
    axs[0].set_ylabel('Predicted Values')

    axs[0].set_title(f'Training Set (MSE: {mse_train:.2f})')
    axs[0].legend()
    
    # Scatter plot for the test set
    axs[1].scatter(y_test, y_pred, alpha=0.5, label='Data')
    
    # Regression line for test set
    m_test, b_test = np.polyfit(y_test, y_pred, 1)
    axs[1].plot(y_test, m_test * y_test + b_test, color='blue', label=f'Fit: y={m_test:.2f}x + {b_test:.2f}')
    
    # Perfect prediction line for reference
    axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    axs[1].set_title(f'Test Set (MSE: {mse_test:.2f})')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
