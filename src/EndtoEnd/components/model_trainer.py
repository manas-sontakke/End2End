import os
import sys
import numpy as np
from dataclasses import dataclass
from urllib.parse import urlparse   

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow

from src.EndtoEnd.exception import CustomException
from src.EndtoEnd.logger import logging

from src.EndtoEnd.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            # For Hyperparameter Tuning, write parameters
            parameters = { 
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Gradient Boosting': {
                    'learning_rate': [.1, .01, .05, .001],      
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Linear Regression': {},
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'K-Neighbors Regressor': {
                    'n_neighbors': [5, 7, 9, 11, 13, 15]
                }, 
                'XGBRegressor': {
                    'learning_rate': [.1, .01, .05, .001],      
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoosting Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100]
                },
                'AdaBoostRegressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [50, 100, 200]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, parameters=parameters)

            # To get the best model score from the dictionary
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best model name: ", best_model_name)

            param_list = list(parameters.keys())

            actual_model = ""

            for model in param_list:
                if model == best_model_name:
                    actual_model = actual_model + model

            best_parameters = parameters[actual_model]


            mlflow.set_registry_uri("https://dagshub.com/sontakke.manas/End2End.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (rmse, mae, r2) = self.evaluate_metrics(y_test, predicted_qualities)
                mlflow.log_params(best_parameters)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)   
                mlflow.log_metric("r2", r2)

                #model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the model registry, which depends on the use case, but for simplicity, we are just using the name of the model as the registered model name
                    # please refer to the doc for more information: https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)  
            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)