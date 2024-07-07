
import numpy as np
import pandas as pd
from sklearn.metrics import (   mean_squared_error,
                                mean_absolute_error,
                                mean_absolute_percentage_error,
                                r2_score,
                                root_mean_squared_error)





class MOD:
    model_metrics=pd.DataFrame()
    
    @staticmethod
    def r2_score_adjusted(y_pred, y_test, X_test):
        r2 = r2_score(y_pred, y_test)
        n = X_test.shape[0]  # number of observations
        p = X_test.shape[1]  # number of observations
        return  1 - (1 - r2) * ((n - 1) / (n - p - 1))

    
    
    @staticmethod
    def get_metrics(vec_, _vec , /):
        
        return {'mean_squared_error':mean_squared_error(vec_, _vec),
                'mean_absolute_error':mean_absolute_error(vec_, _vec),
                'mean_absolute_percentage_error':mean_absolute_percentage_error(vec_, _vec),
                'r2_score':r2_score(vec_, _vec),
                'root_mean_squared_error':root_mean_squared_error(vec_, _vec) }

    @staticmethod
    def approaching_the_model(data):
        mean_line_values=(data.factual+data.model)/2
        
        return pd.Series(map(round,
                                ([np.random.uniform(factual_value, mean_line_value) if indicator else 
                                      factual_value for indicator, factual_value, mean_line_value in zip(
                                                                                      data.indicator_nan,
                                                                                      data.factual,
                                                                                      mean_line_values)])))