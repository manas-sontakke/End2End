import os
import sys

from src.EndtoEnd.exception import CustomException
from src.EndtoEnd.logger import logging
import pandas as pd
import pymysql

from dotenv import load_dotenv

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
