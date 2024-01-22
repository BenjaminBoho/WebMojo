"""
電圧降下の異常検知
パラメーター
移動平均日数：mv_v　初期 8
回帰に使用するデータ期間：days　初期 91

"mv_v"日移動平均をとり、その移動平均のデータ（最長"days"日）を使って1次回帰
傾きが負且つ電圧が"v_lim"未満になれば異常検知”電圧降下”とする。
一度異常検知すればリセットされるまで同じ警報は出さない。
3.55Vを超えたらリセット
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import datetime
import sys
import traceback

import trcom
import daily

warnings.filterwarnings('ignore')

#ファイル名をつけて図を保存
def savefig(prms, brainresult, graphid):
    noimagefile = prms.params["noimagefile"]
    if noimagefile: return
    plt.savefig(prms.workfolder + "/" + graphid + ".png")
    brainresult["files"].append(graphid + ".png")
    return

def battery_trend(prms, brainresult):
    prm = prms.params
    mv_v = prm["mv_v"] #8 #8日移動平均
    v_lim = prm["v_lim"] #3.48
    sensortype = prm["sensortype"]
    sensorid = prm["sensorid"]
    fromdt = prm["fromdt"].replace("/", "-")    #yyyy/mm/dd を yyyy-mm-ddに変換 文字列なので
    unitsystem = prm["unitsystem"]

    brainresult["output"]["battery"] = ""

    #excelの読み込み
    #v_trend=pd.read_excel("C:/Users/H4869/TR-BRAIN/v_trend.xlsx", sheet_name=0, index_col=None, usecols =3,skiprows=None, encoding="cp932")
    #DBからadver_log読み込み
    v_trend = trcom.get_adver(sensortype, sensorid, fromdt)
    #データが無いときは計算できない
    if (len(v_trend) == 0):
        brainresult["output"]["battery"] = "NoAdverData"
        return 
    
    # センサタイプ M は元がips2 単位系をMにするときは m/s2
    # センサタイプ I は元がips 単位系をMにするときは mm/s
    if (sensortype == "M" and unitsystem == "M"):
        coef = trcom.UnitConv.acceleration.get_coef("I", "M")
        v_trend.loc[:,"VRMSXAcc"] = v_trend.loc[:,"VRMSXAcc"] * coef

    #日付をindexにする
    #datetime64[ns]型の列をインデックスに指定すると、そのインデックスはDatetimeIndex型となる。
    v_trend.set_index(["recordDate"], inplace=True)

    #電圧と加速度のトレンドをプロットして影響を確認
    plt.figure(figsize=(10,5)).autofmt_xdate()
    plt.subplot().plot(v_trend["Battery"],color="b",label="Voltage")
    plt.title("Trend of Voltage & ACC")
    plt.xlabel("Date")
    plt.ylabel("Battery Voltage")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

    acc_trend = plt.subplot().twinx() # 第二軸
    acc_trend.plot(v_trend["VRMSXAcc"],color="r", label="ACC RMS")
    plt.ylabel("ACC RMS")

    #plt.show()
    #savefig(prms, "Trend of Voltage & ACC") #TR-BRAINに表示
    savefig(prms, brainresult, "TrendVoltageACC") #TR-BRAINに表示

    #移動平均
    #データ数ではなく、日数で判断
    #最低1週間に1回は点検する（アドバタイズを受ける）として、余裕を見て8日とする。

    mv=str(mv_v) + "D" 

    v_trend["Battery"].rolling(mv).mean()

    #移動平均から線形1次近似

    #目的変数
    #minpval = 3 #最低3つデータが必要
    #v_trend_mv=v_trend["Battery"].rolling(mv,min_periods=minpval).mean().dropna().tail(mv0) #mv0個のデータで計算

    #個数ではなく日数指定でデータ取り出し
    term2 = datetime.timedelta(days=91)#最長約3ヶ月分
    v_trend_mv=v_trend["Battery"].rolling(mv).mean()
    v_trend_mv_term2=v_trend_mv[v_trend_mv.index.max()- term2 : v_trend_mv.index.max()]

    #v_trend_mv=v_trend["Battery"].rolling(mv).mean() #全トレンドデータ考慮の場合

    #説明変数
    #x_mv=np.arange(len(v_trend_mv_term2)).reshape(-1, 1)
    x_mv=v_trend_mv_term2.index.map(pd.Timestamp.timestamp) #日付をそのまま計算に使う。

    #近似式の係数
    #res_mv=np.polyfit(x_mv.reshape(-1), v_trend_mv_term2, 1) #1次
    res_mv=np.polyfit(x_mv, v_trend_mv_term2, 1) #1次

    #近似式の計算
    y_mv = np.poly1d(res_mv)(x_mv)

    #グラフ表示
    plt.figure(figsize=(10, 5))

    plt.subplot()
    plt.plot(v_trend_mv.index, v_trend_mv, label="mv",color="b")

    plt.plot(v_trend["Battery"],linestyle = "dashed", label="org",color="k")
    plt.plot(v_trend_mv_term2.index, y_mv, label="regression",color="r")

    plt.title("Mooving Average {} days of v_trend".format(mv))
    plt.xlabel("Date")
    plt.ylabel("Battery Voltage")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

    #plt.show()
    #savefig(prms, "Mooving Average of v_trend") #TR-BRAINに表示
    savefig(prms, brainresult, "MoovingAveragev_trend") #TR-BRAINに表示

    #傾きを記録
    brainresult["output"]["batteryslope"] = res_mv[0]
    #劣化フラグ 傾きが負のときにtrue
    if res_mv[0] < 0: brainresult["output"]["batterydeterioration"] = True

    #傾きが負で電圧が"v_lim"未満なら電圧降下と判定
    if res_mv[0] < 0 and v_trend_mv[v_trend.index.max()]<v_lim:
        brainresult["output"]["battery"] = "Low"
        print("電圧降下")
    else:
        brainresult["output"]["battery"] = "OK"
        print("正常")
        


#メイン処理
def main():
    params = daily.initcommon()

    if (params.guid == "braintest01"):
        battery_trend(params, params.params["brainresult"])

    params.save_result(params.params["brainresult"])
    return

if __name__=='__main__':
    main()


