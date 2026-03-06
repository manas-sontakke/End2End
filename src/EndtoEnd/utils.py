import os
import sys

from src.EndtoEnd.exception import CustomException
from src.EndtoEnd.logger import logging
import pandas as pd
import numpy as np
import pymysql

from dotenv import load_dotenv

import pickle

load_dotenv() # Load environment variables from .env file

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

def read_sql_data():
    logging.info("Reading data from MySQL database.")
    try:
        mydb = pymysql.connect(host=host, user=user, password=password, database=database)
        logging.info("Connection Established", mydb)
        df=pd.read_sql("SELECT * FROM student", mydb)
        print(df.head())
        return df
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)