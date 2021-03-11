import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
import seaborn as sns
from pydataset import data
from env import host, user, password
import os

def wrangle_telco(host = host, user = user, password = password):
     import pandas as pd
     db = 'telco_churn'
     telco = pd.read_sql('SELECT customer_id, monthly_charges, tenure, total_charges FROM customers WHERE contract_type_id = 2', f'mysql+pymysql://{user}:{password}@{host}/{db}')
     telco['total_charges'] = pd.to_numeric(telco['total_charges'],errors='coerce')
     telco.fillna(0, inplace = True)
     return telco