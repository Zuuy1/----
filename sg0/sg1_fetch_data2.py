import baostock as bs
import pandas as pd
import numpy as np
import os
from pathlib import Path

def rs_to_df(rs):
    dlst=[]
    assert rs is not None
    while (rs.error_code=='0') and rs.next():
        dlst.append(rs.get_row_data())
    return pd.DataFrame(dlst,columns=rs.fields)
#登录
lg=bs.login()
print("lg_error=",lg.error_code,'\n',"lg_msg=",lg.error_msg)

fields = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,"
    "psTTM,pcfNcfTTM,isST"
)
rs=bs.query_hs300_stocks()
code_df=rs_to_df(rs)
test_codes = code_df["code"].head(5).tolist()
#拼路径
project_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(project_dir,"data(11_6try)","raw","prices")
os.makedirs(output_dir,exist_ok=True)
for code in test_codes:
    filename=code.replace('.','_')+".parquet"
    rs = bs.query_history_k_data_plus(
    code=code,
    fields=fields,
    start_date="2020-01-01",
    end_date="2025-12-31",
    frequency="d",
    adjustflag="2"
)
    k_df=rs_to_df(rs)
    output_path=os.path.join(output_dir,filename)
    k_df.to_parquet(output_path, index=False)
    
project_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(project_dir,"data(11_6try)")  
os.makedirs(output_dir, exist_ok=True)
code_df.to_csv(os.path.join(output_dir, "hs300_list.csv"),
               index=False, encoding="utf-8-sig")



#登出baostock
lg=bs.logout()
print('logout')