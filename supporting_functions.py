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

# %% feature engineering and post-processing

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

def qcd_variance(series,window=12): 
    """
    Returns the quartile coefficient of dispersion of the 
    rolling variance of a series in a given window.

    Args:
        series (pandas.DataFrame): Time series dataframe
        window (int, optional): Size of rolling variance window. Defaults to 12. 
    """
    # rolling variance for a given window 
    variances = series.rolling(window).var().dropna()
    # first quartile
    Q1 = np.percentile(variances, 25, interpolation='midpoint')
    # third quartile
    Q3 = np.percentile(variances, 75, interpolation='midpoint')
    # quartile coefficient of dispersion 
    qcd = round((Q3-Q1)/(Q3+Q1),6)
    
    print(f"quartile coefficient of dispersion: {qcd}")

# %% plotting
  
def prediction_plot(y_test, y_pred):
    """
    Evaluate the model by plotting scatter plots of true vs predicted values
    with a least squares regression line and MSE.
    
    Parameters:
    y_test (array-like): True values for the test set.
    y_pred (array-like): Predicted values for the test set.
    """
    plt.figure(figsize=(6, 5))
    mse_test  = mean_squared_error(y_test,  y_pred)
      
    # Scatter plot for the test set
    plt.scatter(y_test, y_pred, alpha=0.5, label='Data')
    
    # Regression line for test set
    m_test, b_test = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m_test * y_test + b_test, color='blue', label=f'Fit: y={m_test:.2f}x + {b_test:.2f}')
    
    # Perfect prediction line for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test Set (MSE: {mse_test:.2f})')
    plt.legend()
    
    plt.tight_layout()

def plot_permutation_importance(feature_importance,labels):
    """Make boxplots of feature importance.

    Args:
        feature_importance (dict): Dictionary of feature importances.
        labels (list): Names of the features.
    """    
    _, ax = plt.subplots(figsize=(7, 6))
    perm_sorted_idx = feature_importance.importances_mean.argsort()

    ax.boxplot(
        feature_importance.importances[perm_sorted_idx].T,
        vert=False,
        labels=labels[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
