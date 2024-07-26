import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

import pandas as pd


if __name__ == "__main__":

    # Before being able to load data, you need a .env file with the following info:

    #  For psycopg2: replace values for USERNAME and PASSWORD with your own login info
    # ```
    # DATABASE = "postgres"
    # USER_DB = "USERNAME"
    # PASSWORD = "PASSWORD"
    # HOST = "hostlink"
    # PORT = "portnumber"
    # ```

    #  For sqlalchemy: replace "USERNAME" and "PASSWORD" with your own
    # ```
    # DB_STRING = "DATABASEsql://USERNAME:PASSWORD@hostlink"


    #make data folder if needed
    if not os.path.exists("./data"):
            os.mkdir("data")
    
    #read database string from .env file (no need to change anything)
    load_dotenv()
    DB_STRING = os.getenv('DB_STRING')
    db = create_engine(DB_STRING)

    #define SQL query to download data 
    query_string = """
    SET SCHEMA 'xyz';
    SELECT 
        *
    FROM 
        table_1
    LEFT JOIN 
        table_2 ON table_1.private_key = table_2.foreign_key;
    """

    #import with pandas
    df = pd.read_sql(query_string, db)

    #save to .csv file
    df.to_csv("data/my_data.csv", index=False) #save dataframe as .csv file