import pandas as pd
import numpy as np
import baostock as bs
from pandas import DataFrame
from datetime import date,time,datetime,timedelta
import os

def into_rs(rs):
    rows=[]
    while(rs.error_code=="0" and rs.next()):
        rows.append(rs.get_row_data())
    return DataFrame(rows,columns=rs.fields)

# lg=bs.login()
# print("login:",lg.error_code,lg.error_msg)

# rs = bs.query_hs300_stocks()
# hs300 = into_rs(rs)

# print(hs300.head(20))

# bs.logout()

save_ph="output"
k_path=os.path.join(save_ph,"sh600000_kdata_2023.csv")
try_pandas=pd.read_csv(k_path) 
print(try_pandas.head())

