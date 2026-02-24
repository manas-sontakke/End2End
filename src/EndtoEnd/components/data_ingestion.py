# Database/Data-Source (MySQL) -> Data -> Train/Test Split (2 Output Files) => Input required Train/Test File Paths

import os
import sys

from src.EndtoEnd.exception import CustomException
from src.EndtoEnd.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.EndtoEnd.utils import read_sql_data #Important
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## Reading data from MySQL Database
            df = read_sql_data()
            logging.info("Reading from MySQL database completed.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)