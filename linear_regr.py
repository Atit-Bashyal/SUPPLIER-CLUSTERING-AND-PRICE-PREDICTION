#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:22:39 2019

@author: Atit Bashyal and Peeter Mansoor

py file to import linear regression model with MSE as loss function
"""

from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class linear_regr:

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)
