import pandas as pd
import numpy as np

df=pd.read_csv('C:\\old system\\customer intermediate\\final 3\\customer_purchase_intermediate\\data\\customers.csv')
print(df.info())
print(df.describe())
print(df.isnull().sum())

from ydata_profiling import ProfileReport
profile=ProfileReport(df,title="customer analysis report",explorative=True)
profile.to_file("analysis_report.html")