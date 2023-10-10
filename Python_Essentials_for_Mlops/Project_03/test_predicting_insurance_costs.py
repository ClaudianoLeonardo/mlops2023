import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from predicting_insurance_costs import load_data, transform_data, train_linear_regression, evaluate_model
import pytest
def test_load_data():
    data = load_data("Python_Essentials_for_Mlops/Project_03/data/insurance.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_transform_data():
    test_data = pd.DataFrame({
        'age': [30, 40, 50],
        'bmi': [25, 30, 35],
        'charges': [3000, 4000, 5000],
        'smoker': ['yes', 'no', 'yes']
    })

    X, y = transform_data(test_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 3
    assert y.shape[0] == 3

def test_train_linear_regression():
    X_train = pd.DataFrame({
        'age': [30, 40, 50],
        'bmi': [25, 30, 35],
        'is_smoker': [1, 0, 1]
    })
    y_train = pd.Series([3000, 4000, 5000])

    model = train_linear_regression(X_train, y_train)
    assert isinstance(model, LinearRegression)

def test_evaluate_model():
    model = LinearRegression()

    X = pd.DataFrame({
        'age': [30, 40, 50],
        'bmi': [25, 30, 35],
        'is_smoker': [1, 0, 1]
    })
    y = pd.Series([3000, 4000, 5000])

    model = train_linear_regression(X, y)
    assert isinstance(model, LinearRegression)

    mse, r2 = evaluate_model(model, X, y)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
