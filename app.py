from src.EndtoEnd.logger import logging
from src.EndtoEnd.exception import CustomException
from src.EndtoEnd.components.data_ingestion import DataIngestion
from src.EndtoEnd.components.data_ingestion import DataIngestionConfig
from src.EndtoEnd.components.data_transformation import DataTransformationConfig, DataTransformation
import sys

if __name__ == "__main__":
    logging.info("The execution of EndtoEnd has started.")

    try:
        #data_ingestion = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        data_ingestion.initiate_data_ingestion()
        
        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        
        data_transformation.initiate_data_transformation(train_path="artifacts/train.csv", test_path="artifacts/test.csv")

    except Exception as e:
        logging.info("Custom exception has been raised.")
        raise CustomException(e, sys) from e