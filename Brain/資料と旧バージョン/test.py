import pandas as pd
import trcom

stmt = "select * from trendshourly where sensortype=%s and sensorid=%s and record_date >= %s and record_date < %s;"
prms = ("M", "VB377428", "2018/11/01 00:00:00", "2020/03/31 00:00:00",)
df = pd.read_sql_query(stmt, trcom.dbcon(), params=prms)
print(df)
