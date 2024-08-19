# Predictive modeling skeletonscripts
Bare bones skeleton scripts for predictive modeling of cross-sectional data. The aim of these scripts is to have a set of easily adjustable templates for loading data and fitting predictive models. A few examples of adjusted templates for running basic model fits is also included.

`example_fit_linear_classification_model.ipynb`, cleans data from a dataset about titanic passenger information, runs simple feature engineering, and fits a logistic regression model to predict if a passenger survived the sinking. It is not intended to showcase the best model for this purpose (the model does alright but other models can do better) or how to best construct a regression model. Instead, the aim is to show how with relatively few modifications, we can turn `skeleton_fit_model` into a functional pipeline for a classification problem.

List of files:
- `load_data.ipynb`: jupyter notebook for loading and saving datasets (titanic and iris)
- `load_data.py`: same as above but python script
- `skeleton_load_data_SQL.ipynb`: template notebook for loading data from postgress database with an SQL query
- `skeleton_load_data_SQL.py`: same as above but python script
- `skeleton_fit_model.ipynb`: template notebook for modeling pipeline
- `skeleton_fit_model.py`: same as above but python script
- `example_fit_linear_classification_model.ipynb`: example of modified template to produce functional pipeline for classification problems
- `example_fit_nonlinear_classification_model.ipynb`: another example, this time with a non-linear model (xgboost)
- `example_fit_regression_model.ipynb`: example of modified template to produce functional pipeline for regression problems
- `supporting_functions.py`: file that contains helper functions for data cleaning and feature filtering

### **Installation, for `macOS`**: 


- Install the virtual environment and the required packages:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **Installation, for `WindowsOS`**:

- Install the virtual environment and the required packages:

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :

  ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```