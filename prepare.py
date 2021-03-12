import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date
import seaborn as sns
from pydataset import data
from env import host, user, password
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

def scale_prep(train, validate, test):
    scaler_minmax = sklearn.preprocessing.MinMaxScaler()
    scaler_minmax.fit(train)
    train = scaler_minmax.transform(train)
    train = pd.DataFrame(train)
    train.rename(columns = {0: 'monthly_charges', 1: 'tenure', 2: 'total_charges'}, inplace = True)
    validate = scaler_minmax.transform(validate)
    validate = pd.DataFrame(validate)
    validate.rename(columns = {0: 'monthly_charges', 1: 'tenure', 2: 'total_charges'}, inplace = True)
    test = scaler_minmax.transform(test)
    test = pd.DataFrame(test)
    test.rename(columns = {0: 'monthly_charges', 1: 'tenure', 2: 'total_charges'}, inplace = True)
    return train, test, validate