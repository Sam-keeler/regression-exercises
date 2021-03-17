from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date
import seaborn as sns
from pydataset import data
from env import host, user, password
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from math import sqrt

def plot_residuals(y, db, x):
    model = ols('y ~ x', data=db).fit()
    predictions = model.predict(x)
    residuals = y - predictions
    return sns.scatterplot(x, residuals) 

def regression_errors(y, yhat, db, x):
    model = ols('{y} ~ {yhat}', data=db).fit()
    predictions = model.predict(db.x)
    SSE = sum((y - predictions)**2)
    MSE = SSE/len(db)
    RMSE = sqrt(MSE)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    #return "SSE =" SSE, "MSE =" MSE, "RMSE =" RMSE, "ESS =" ESS, "TSS =" TSS

def baseline_mean_errors(y):
    SSE_baseline = sum((y - y.mean())**2)
    MSE_baseline = SSE_baseline/len(y)
    RMSE_baseline = sqrt(MSE_baseline)
    #return "SSE_baseline =" SSE_baseline, "MSE_baseline =" MSE_baseline, "RMSE_baseline =" RMSE_baseline

def better_than_baseline(y, yhat):
    SSE = sum((y - predictions)**2)
    SSE_baseline = sum((y - y.mean())**2)
    print("SSE =", round(SSE, 2))
    print("SSE_baseline =", round(SSE_baseline, 2))
    if SSE > SSE_baseline:
        print("The baseline has performed better than the model")
    else:
        print("The model has outperformed the baseline")