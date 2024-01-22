# しきい値の自動提案
import pandas as pd
import datetime
import statistics
import statsmodels.api as sm
from scipy import stats

def suggest_threshold(df, prms):
    prm = prms.params
    brainresult = prm["brainresult"]
    minval = prm["minval"]
    countstartdate = prm["countstartdate"]      # 取り付け日
    todt = prm["todt"]                  # 取得最終日 通常起動日時 
    tholdcount = prm["tholdcount"]      # X1 46
    tholddays = prm["tholddays"]        # X2 23日
    apercent = prm["apercent"]          # a% 50%
    tholdlow = prm["tholdlow"]          # 注意しきい値
    tholdhigh = prm["tholdhigh"]        # 異常しきい値
    tholdnewcoeflow = prm["tholdnewcoeflow"]    # 提案注意しきい値係数
    tholdnewcoefhigh = prm["tholdnewcoefhigh"]  # 提案異常しきい値係数
    timezoneoffsetprm = prm["timezoneoffset"]

    # 処理を行うかどうか カウンタ開始日時からの日数で判断する
    tmfmt = '%Y/%m/%d %H:%M:%S'

    todttime = datetime.datetime.strptime(todt, tmfmt)
    elapsedt = datetime.datetime.strptime(todt, tmfmt) - datetime.datetime.strptime(countstartdate, tmfmt)
    if elapsedt.days < 30:
        brainresult["output"]["tholdmessages"].append("カウンタ開始からの期間が短い 日数 {}".format(elapsedt.days))
        return

    #タイムゾーンオフセットをtimedeltaフォーマットにしておく
    timezoneoffset = datetime.timedelta(hours=timezoneoffsetprm)
    if (minval == None):
        minval = 1  #1.0 #加速度1.0m/s2
    #判断しきい値 tholdlow
    if (tholdlow == None):
        if (tholdhigh == None):
            return
        else:
            tholdlow = tholdhigh

    #取得日開始日
    tholdfromdt = todttime + timezoneoffset - datetime.timedelta(days=tholddays)
    tholdfromdt = tholdfromdt.replace(hour=4, minute=0, second=0) # 日付の00:00:00にしておく

    #期間対象データ
    df_x2 = df[tholdfromdt:]
    if df_x2.size < tholdcount:
        brainresult["output"]["tholdmessages"].append("期間対象要素数が少ない 個数{}".format(df_x2.size))
        return

    #注意しきい値以上のデータ
    df_x2_filter = df_x2[df.loc[:, 'y'] > tholdlow]

    #条件1
    percent = df_x2_filter.size / df_x2.size * 100
    if percent <= apercent:
        brainresult["output"]["tholdmessages"].append("条件1を満たしていない %={}".format(percent))
        return
    print(percent)

    #条件2
    df_all = df[df.loc[:, 'y'] > minval] #起動検知しきい値以上のデータ
    if df_all.size < 3:
        brainresult["output"]["tholdmessages"].append("条件2 起動検知しきい値を超えた要素数が少ない {}".format(df_all.size))
        return
    median = statistics.median(df_all['y'])
    print(median)
    overmedian = df_all[df.loc[:, 'y'] > median]
    if overmedian.size < 3:
        brainresult["output"]["tholdmessages"].append("条件2 中央値を超えた要素数が少ない中央値={} 数={}".format(median, overmedian.size))
        return
    acf = sm.tsa.stattools.acf(overmedian, nlags=int(len(overmedian)/3))
    kurtosis = stats.kurtosis(acf)
    print(kurtosis)
    if (kurtosis <= 0):
        brainresult["output"]["tholdmessages"].append("条件2 尖り度が0以下 尖り度={}".format(kurtosis))
        return

    #提案しきい値の計算
    median_x2 = statistics.median(df_x2[df.loc[:, 'y'] > minval]['y']) #起動検知しきい値以上のデータから計算
    overmedian_x2 = df_x2[df_x2.loc[:, 'y'] > median_x2]
    if overmedian_x2.size < 1:
        brainresult["output"]["tholdmessages"].append("提案しきい値の計算 中央値超えているものが1個以下 中央値={}".format(median_x2))
        return
    desc_x2 = overmedian_x2['y'].describe()
    ave_x2 = desc_x2.loc['mean']
    stddev_x2 = desc_x2.loc['std']
    print(ave_x2)
    print(stddev_x2)
    brainresult["output"]["suggesttholdlow"] = ave_x2 + stddev_x2 * tholdnewcoeflow
    brainresult["output"]["suggesttholdhigh"] = ave_x2 + stddev_x2 * tholdnewcoefhigh   
    return

def main():
    #テストデータ
    date_ones = pd.DataFrame(
    index=[
        pd.Timestamp('2021-12-01 03:00:00'),
        pd.Timestamp('2021-12-02 03:00:00'),
        pd.Timestamp('2021-12-03 03:00:00'),
        pd.Timestamp('2021-12-04 03:00:00'),
        pd.Timestamp('2021-12-05 03:00:00'),
        pd.Timestamp('2021-12-06 03:00:00'),
        pd.Timestamp('2021-12-07 03:00:00')
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
    date_ones

    #テスト用パラメータ
    class prms:
        params = {"minval":1,"brainresult":{}}
    
    prm = prms.params
    brainresult = {"files":[], "output":{}}
    brainresult["output"]["tholdmessages"] = []
    prm["brainresult"] = brainresult
    prm["timezoneoffset"] = 9
    prm["countstartdate"] = "2021/12/01 12:00:00"
    prm["todt"] = "2021/12/20 12:00:00"
    prm["tholdcount"] = 3   #46
    prm["tholddays"] = 23   #23日
    prm["apercent"] = 50    #50%
    prm["minval"] = 1       # 起動検知しきい値
    prm["tholdlow"] = 2     # 注意しきい値
    prm["tholdhigh"] = 3    # 異常しきい値
    prm["tholdnewcoeflow"] = 4    # 提案注意しきい値係数
    prm["tholdnewcoefhigh"] = 6   # 提案異常しきい値係数

    # テスト呼び出し    
    suggest_threshold(date_ones, prms)
    print(prm["brainresult"]["output"])
    return

if __name__=='__main__':
    main()


