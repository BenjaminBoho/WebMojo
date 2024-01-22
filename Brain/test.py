import pandas as pd
import datetime

# df = pd.DataFrame({'value': range(1, 32, 2)},
#                   index=pd.date_range('2018-01-01', '2018-01-31', freq='2D'))


df = pd.DataFrame(
index=[
    pd.Timestamp('2022-10-01 03:00:00'),
    pd.Timestamp('2022-10-02 03:00:00'),
    pd.Timestamp('2022-10-03 03:00:00'),
    pd.Timestamp('2022-10-04 03:00:00'),
    pd.Timestamp('2022-10-05 03:00:00'),
    pd.Timestamp('2022-10-06 03:00:00'),
    pd.Timestamp('2022-10-09 03:00:00')
    ],
    data=[
        0.1,
        0.2,
        2,
        3.2,
        5,
        0.3,
        0.2]
    ,columns=["y"]
)                  
s = pd.DataFrame(df.iloc[:,0].rolling('4D')).tail(1).T

print(s)

# tmfmt = '%Y/%m/%d %H:%M:%S'
# dif = datetime.datetime.now() - datetime.datetime.strptime("2022/10/13 17:29:00", tmfmt)
# if dif.total_seconds() < (24 * 60 * 60):
#     print(dif.total_seconds())
# else:
#     print("たってる")
