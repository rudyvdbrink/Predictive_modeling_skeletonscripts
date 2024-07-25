import pandas as pd
import os
from sklearn.datasets import load_iris
import seaborn as sns

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("data")

    #get iris dataset
    iris = load_iris()

    df = pd.DataFrame(
        iris.data, 
        columns=iris.feature_names
        )

    df['target'] = iris.target

    # Map targets to target names
    target_names = {
        0:'setosa',
        1:'versicolor', 
        2:'virginica'
    }

    df['target_names'] = df['target'].map(target_names)

    #clean up column names
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('(',  '') for col in df.columns]
    df.columns = [col.replace(')',  '') for col in df.columns]

    df.to_csv('data/data_iris.csv', sep='\t',index=False)
    print("Iris data saved successfully")

    #get titanic dataset
    df = sns.load_dataset('titanic')
    df.to_csv('data/data_titanic.csv', sep='\t',index=False)
    print("Titanic data saved successfully")

    exit()