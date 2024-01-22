# しきい値の自動提案
from ast import IsNot
# from asyncio.windows_events import NULL
import pandas as pd
import datetime
import statistics
import statsmodels.api as sm
from scipy import stats

def rising_trend_alarm(df, prms):
    prm = prms.params
    brainresult = prm["brainresult"]
    minval = prm["minval"]
    todt = prm["todt"]                  # 取得最終日 通常起動日時 
    tholdcount = prm["tholdcount"]      # X1 46
    tholddays = prm["tholddays"]        # X2 23日
    tholdnewcoeflow = prm["tholdnewcoeflow"]    # 提案注意しきい値係数
    timezoneoffsetprm = prm["timezoneoffset"]
    trendrisingalarmthold = prm["trendrisingalarmthold"]            # トレンド上昇アラームしきい値
    trendrisingalarmtholddate = prm["trendrisingalarmtholddate"]    # トレンド上昇アラーム設定最終日時
    trendrisingalarmstatus = prm["trendrisingalarmstatus"]          # トレンド上昇アラームステータス
    trendrisingalarmNGdate = prm["trendrisingalarmNGdate"]          # トレンド上昇アラームNG日時

    tmfmt = '%Y/%m/%d %H:%M:%S'
    todttime = datetime.datetime.strptime(todt, tmfmt)
    # トレンド上昇アラームしきい値がnullでないとき(2回め以降)
    if trendrisingalarmthold is not None:
        elapsedt = todttime - datetime.datetime.strptime(trendrisingalarmtholddate, tmfmt)
        if elapsedt.days < 30:
            if trendrisingalarmstatus != "NG":
                brainresult["output"]["tholdmessages"].append("トレンド上昇アラームステータスがNGでない")
                return
            statusngelapedt = todttime - datetime.datetime.strptime(trendrisingalarmNGdate, tmfmt)
            if statusngelapedt.total_seconds() > (24 * 60 * 60):
                brainresult["output"]["tholdmessages"].append("トレンド上昇アラームNG日時から現在日時が24時間たっている")
                return
 
    #タイムゾーンオフセットをtimedeltaフォーマットにしておく
    timezoneoffset = datetime.timedelta(hours=timezoneoffsetprm)

    exist30dt = todttime + timezoneoffset - datetime.timedelta(days=30)
    if df[exist30dt:].size < 1:
        brainresult["output"]["tholdmessages"].append("トレンド上昇アラーム 現在より30日以内のデータがない")
        return

    if (minval == None):
        minval = 1  #1.0 #加速度1.0m/s2

    #取得日開始日
    x23 = '{}D'.format(tholddays)
    # 起動検知しきい値以上のデータ
    df_filtered = df[df.iloc[:,0]> minval]

    # 641_データ範囲の変更（トレンド上昇傾向アラーム）により削除
    # 窓関数を使ってx23日毎の最後の窓のデータを取得 (対象日以外はNaNとなるので除外)
    # df_x2 = pd.DataFrame(df_filtered.loc[:,'y'].rolling(x23)).tail(1).T.dropna()

    # 641_データ範囲の変更（トレンド上昇傾向アラーム）
    # 起動検知しきい値以下で削除された日数を除いた正味の23日分としたい。
    if (df_filtered.size < 1) :
        brainresult["output"]["tholdmessages"].append("トレンド上昇アラーム 起動検知しきい値フィルダー後の要素数が少ない 個数{}".format(df_filtered.size))        
        return
    startdate = df_filtered.groupby(pd.Grouper(freq="D")).mean().dropna().tail(tholddays).index[0]
    df_x2=df_filtered[startdate:]

    if df_x2.size < tholdcount:
        brainresult["output"]["tholdmessages"].append("トレンド上昇アラーム 直近X2期間中に、起動検知しきい値フィルダー後の要素数が少ない 個数{}".format(df_x2.size))
        return

    #提案しきい値の計算
    median_x2 = statistics.median(df_x2.loc[:, 'y']) 
    overmedian_x2 = df_x2[df_x2.loc[:, 'y'] > median_x2]
    if overmedian_x2.size < 2:
        brainresult["output"]["tholdmessages"].append("トレンド上昇アラーム 中央値超えているものが2個より少ない 中央値={}".format(median_x2))
        return

    desc_x2 = overmedian_x2['y'].describe()
    ave_x2 = desc_x2.loc['mean']
    stddev_x2 = desc_x2.loc['std']
    print(ave_x2)
    print(stddev_x2)
    brainresult["output"]["trendrisingalarmthold"] = ave_x2 + stddev_x2 * tholdnewcoeflow
    return

def main():
    #テストデータ
    date_ones = pd.DataFrame(
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
    date_ones

    #テスト用パラメータ
    class prms:
        params = {"minval":1,"brainresult":{}}
    
    prm = prms.params
    brainresult = {"files":[], "output":{}}
    brainresult["output"]["tholdmessages"] = []
    prm["brainresult"] = brainresult
    prm["timezoneoffset"] = 9
    prm["todt"] = "2022/10/31 12:00:00"
    prm["tholdcount"] = 3   #46
    prm["tholddays"] = 23   #23日
    prm["minval"] = 0.1       # 起動検知しきい値
    prm["tholdnewcoeflow"] = 4    # 提案注意しきい値係数
    prm["trendrisingalarmthold"] = None                     # トレンド上昇アラームしきい値
    prm["trendrisingalarmtholddate"] = "2021/12/01 12:00:00" # トレンド上昇アラーム設定最終日時
    prm["trendrisingalarmstatus"] = "NG"         # トレンド上昇アラームステータス
    prm["trendrisingalarmNGdate"] = "2022/10/01 12:00:00"         # トレンド上昇アラームNG日時

    # テスト呼び出し    
    rising_trend_alarm(date_ones, prms)
    print(prm["brainresult"]["output"])
    return

if __name__=='__main__':
    main()


