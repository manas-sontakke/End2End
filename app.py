from src.EndtoEnd.logger import logging
from src.EndtoEnd.exception import CustomException
import sys

if __name__ == "__main__":
    logging.info("The execution of EndtoEnd has started.")

    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Custom exception has been raised.")
        raise CustomException(e, sys) from e