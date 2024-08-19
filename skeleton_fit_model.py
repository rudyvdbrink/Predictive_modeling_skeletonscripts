#%% import libraries
#import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import clear_output

from scipy.stats import loguniform

from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

#import supporting functions used for cleaning
from supporting_functions import check_duplicates, one_hot, add_full_interactions, add_interactions, sigmoid, linmap, filter_features, plot_permutation_importance, prediction_plot

#%%  main function for cleaning
def clean_data(df,target_column):
    ### main preprocessing function:

    ## done on train and test set together
    
    #clean up column names
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('(',  '') for col in df.columns]
    df.columns = [col.replace(')',  '') for col in df.columns]
    df.columns = [col.lower() for col in df.columns]

    #drop column names that we don't need
    df = df.drop(['column_name_1', 'column_name_2'],axis=1)

    #check and report on duplicate data
    df = check_duplicates(df)

    #run train-test split
    #note: X still contains the y-variable in the 'target' column, this is because it 
    #easier to remove rows / apply cleaning steps without having to do it separately 
    #for the target data vector.
    X = df
    X_train, X_test, _, _ = train_test_split(X, df[target_column], random_state=1234) 

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)


    ## done separately for train and test

    #imputing
    column_to_impute = 'column_name'
    X_train[column_to_impute] = X_train[column_to_impute].fillna(X_train[column_to_impute].median())
    X_test[column_to_impute]  = X_test[column_to_impute].fillna(X_test[column_to_impute].median())

    #non-linear scaling of values
    X_train['column_name'] = X_train.fare.apply(lambda x: np.log(x) if x > 0 else 0)
    X_test['column_name']  = X_test.fare.apply(lambda x: np.log(x) if x > 0 else 0)

    #define / identify columns for range normalization
    df['column_name'] = df['column_name'].astype('category')
    columns_to_scale = df.select_dtypes(include='number').columns.drop(target_column) #identify the numberic columns
  
    #remove outliers from the training set
    outlier_threshold = X_train['column_name'].median()+(X_train['column_name'].std()*3)
    print(str(np.sum(X_train['column_name'] > outlier_threshold)) + " outliers detected")
    X_train = X_train[X_train['column_name'] < outlier_threshold]
    X_train.reset_index(drop=True, inplace=True)   

    #range normalization
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled  = scaler.transform(X_test[columns_to_scale])
    X_train = pd.concat([X_train_scaled, X_train.drop(columns_to_scale,axis=1)], axis=1)
    X_test  = pd.concat([X_test_scaled, X_test.drop(columns_to_scale,axis=1)], axis=1)

    #separate the target
    y_train = X_train.pop(target_column)
    y_test  = X_test.pop(target_column)

    return X_train, X_test, y_train, y_test

#%%  main function for feature engineering
def feature_engineer(df,columns_to_interact,columns_to_dummycode):
    ### main feature engineering function:

    #add interaction terms
    df = add_interactions(df,columns_to_interact)

    #one-hot encode categorical variables
    df = one_hot(df,columns_to_dummycode)

    return df

#%% main code
if __name__ == "__main__":
    #load data
    df = pd.read_csv('file_name.csv',sep='\t')
    
    #user-defined target for classification
    target_column = 'target'

    #initial exploration
    df.head()
    df.isnull().mean()
    df.info()

    # Checking for data imbalance
    df[target_column].value_counts()

    #%% clean data and make features
    X_train, X_test, y_train, y_test = clean_data(df,target_column)

    sns.pairplot(pd.concat([X_train, y_train],axis=1), hue=target_column, height=2)

    columns_to_interact  = ['column_1','column_2']
    columns_to_dummycode = ['column_3','column_4']

    X_train = feature_engineer(X_train,columns_to_interact,columns_to_dummycode)
    X_test  = feature_engineer(X_test,columns_to_interact,columns_to_dummycode)

    #%% check model
    #select features with filtering
    thresh = 0.95 #remove features that correlate above this threshold
    X_train, X_test = filter_features(X_train,X_test,thresh)

    #plot correlation between regressors
    sns.heatmap(X_train.corr(),vmin=-1,vmax=1)

    #%% fit model (classification)

    # instantiate logistic regression
    lr = LogisticRegression(max_iter=10000)
    #lr.get_params()
    #lr.get_params().keys()

    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1234)

    # define search space
    param_grid = { "solver" : ['newton-cg', 'lbfgs', 'liblinear'],
                "penalty" : ['none', 'l1', 'l2', 'elasticnet'],
                "C" : loguniform(1e-5, 100)}

    # define Random search
    Random_search = RandomizedSearchCV(lr, param_grid, n_iter=500, scoring='accuracy', n_jobs=1, cv=cv, random_state=1)

    # execute Random search
    Random_search.fit(X_train, y_train)

    #make predictions using the trained model
    y_pred_train = Random_search.predict(X_train)
    y_pred       = Random_search.predict(X_test)

    clear_output(wait=False) #remove all the warnings, we don't need them here

    # %% fit model (regression)

    # Initialize the LinearRegression model
    mdl = ElasticNet()

    # Define the parameter grid (in this case, there are no hyperparameters for LinearRegression, so we use an empty grid)
    param_grid = {
                'alpha': [0.01, 0.1, 1, 10],  # Regularization strength
                'l1_ratio': [0.1, 0.5, 0.9]
                }

    #define cross-validation scheme
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1234)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=mdl, 
                            param_grid=param_grid, 
                            cv=5, 
                            scoring='neg_mean_squared_error')

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    #%% evaluate model (classification)
    cfm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cfm, cmap='inferno', annot=True, fmt='d', linewidths=.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_pred))

    #calculate probabilities
    y_prob_train = Random_search.predict_proba(X_train)
    y_prob_test =  Random_search.predict_proba(X_test)

    #calculate baseline and model ROC curves
    base_fpr,  base_tpr,  _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, y_prob_test[:,1])
        
    #plot ROC curves
    plt.figure(figsize = (6, 5))
    plt.plot(base_fpr, base_tpr, 'k', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('ROC curve')
    plt.show()

    print(f'Train ROC AUC Score: {roc_auc_score(y_train, y_prob_train[:,1])}')
    print(f'Test ROC AUC  Score: {roc_auc_score(y_test,  y_prob_test[:,1])}')

    #compute and plot feature importance
    feature_importance = permutation_importance(Random_search, X_test,y_test,n_repeats=10,random_state=1234)
    plot_permutation_importance(feature_importance,X_test.columns)

    # %% evaluate the model (regression)

    # Calculate residuals
    residuals = y_test - y_pred

    # Plot residuals
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals, alpha=0.75)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (log)')
    plt.ylabel('Residuals (log)')
    plt.title('Residual Plot')
    plt.show()

    # Print the best parameters and the corresponding score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", -grid_search.best_score_)

    # Evaluate the model on the test set
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Test Set Mean Squared Error:", mse)
    print("Test Set R^2 Score:", r2)
    plt.show()

    #plot predicted versus true test set values
    prediction_plot(y_test,y_pred)

    #compute and plot feature importance
    feature_importance = permutation_importance(grid_search, X_test,y_test,n_repeats=100,random_state=1234)
    plot_permutation_importance(feature_importance,X_test.columns)