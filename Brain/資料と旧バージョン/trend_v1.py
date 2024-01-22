import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import numpy as np
import pandas as pd
import sys
import time
import json
import traceback

import trcom

brainresult = {"status":False, "files":[], "output":{}}
brainresult["output"]["lifetime1"] = -1
brainresult["output"]["lifetime2"] = -1

trcom.font_setup()
minvaltacc = 1.0 #加速度1.0m/s2
minvalspeed = 0.3 #速度0.3mm/s

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#ファイル名をつけて図を保存
def savefig(prms, graphid):
    plt.savefig(prms.workfolder + "/" + graphid + ".png")
    brainresult["files"].append(graphid + ".png")
    return

#日付軸のグラフを作成
def pltdategraph(prms, graphid, df, dispunit):
    fig = plt.figure()
    ax = plt.subplot()
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
    # plt.gcf().autofmt_xdate() 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.plot(df)
    ax.set_ylabel(dispunit)
    #plt.title("日本語")
    savefig(prms, graphid)
    return

#シリーズのヒストグラム
def plthistgraph(prms, graphid, ds):
    plt.figure()
    ax = plt.subplot()
    ax.hist(ds, bins=30)
    savefig(prms, graphid)
    return

#トレンドデータ
def trend_plot(prms):
    prm = prms.params
    #パラメータ
    sensortype = prm["sensortype"]
    sensorid = prm["sensorid"]
    fromdt = prm["fromdt"]
    todt = prm["todt"]
    axis = prm["axis"]
    unitsystem = prm["unitsystem"]
    minval = prm["minval"]
    maxval = prm["maxval"]
    tholdlow = prm["tholdlow"]
    tholdhigh = prm["tholdhigh"]
    movingdays1 = prm["movingdays1"]
    movingdays2 = prm["movingdays2"]
    movingdays3 = prm["movingdays3"]

    #データの単位
    dispunit = ""
    if (sensortype == "M"):
        if  (unitsystem == "E"):
            dispunit = trcom.UnitConv.acceleration.unitname("I")
            if (minval == None):
                minval = trcom.UnitConv.acceleration.conv("M", "I", minvaltacc)
        else:
            dispunit = trcom.UnitConv.acceleration.unitname("M")    
            if (minval == None):
                minval = minvaltacc
    if (sensortype == "I"):
        if  (unitsystem == "E"):
            dispunit = trcom.UnitConv.speed.unitname("I")
            if (minval == None):
                minval = trcom.UnitConv.speed.conv("MM", "I", minvalspeed)
        else:
            dispunit = trcom.UnitConv.speed.unitname("MM")
            if (minval == None):
                minval = minvalspeed

    results = trcom.get_trend(sensortype, sensorid, fromdt, todt)
    if len(results) == 0:
        brainresult["error"] = "DataNotFound"
        return False

    #Y軸データを取得
    if (sensortype == "M"):
        df = results[["record_date", "x_vrmsacc"]]
    else :
        if (axis == "X"):
            df = results[["record_date", "x_vrms"]]
        if (axis == "Y"):
            df = results[["record_date", "y_vrms"]]
        if (axis == "Z"):
            df = results[["record_date", "z_vrms"]]

    df.columns = ["record_date", "y"]
    #単位変換
    # センサタイプ M は元がips2 単位系をMにするときは m/s2
    # センサタイプ I は元がips 単位系をMにするときは mm/s
    if (sensortype == "M" and unitsystem == "M"):
        coef = trcom.UnitConv.acceleration.get_coef("I", "M")
        df.loc[:,"y"] = df.loc[:,"y"] * coef
    elif (sensortype == "I" and unitsystem == "M"):
        coef2 = trcom.UnitConv.speed.get_coef("I", "MM")
        df.loc[:,"y"] = df.loc[:,"y"] * coef2
    df.set_index("record_date", inplace=True)

    # 原系列
    pltdategraph(prms, "org", df, dispunit)

    # 原系列ヒストグラム
    plthistgraph(prms, "orghist" ,df["y"])

    #しきい値によるフィルター
    df_filtered= df[df["y"] > minval]
    if (maxval != None):
        df_filtered = df[df["y"] < maxval]
    if len(df_filtered) == 0:
        brainresult["error"] = "NoFilteredData"
        return False

    pltdategraph(prms, "filtered", df_filtered, dispunit)
    plthistgraph(prms, "filtered_hist" , df_filtered["y"])

    #LOFによる外れ値検査
    from sklearn.neighbors import LocalOutlierFactor

    #arrayに変換
    array_df_filtered= np.array(df_filtered["y"].values.reshape(-1, 1))
    array_date = np.array(df_filtered.index.to_pydatetime().reshape(-1, 1))

    neinum = len(array_df_filtered)//3 #全データの1/3 (=3つの運転モード)
    if (neinum <= 0):
        brainresult["error"] = "NotEnoughDataToTarget"
        return False

    lof = LocalOutlierFactor(n_neighbors=neinum) 
    pred=lof.fit_predict(array_df_filtered)
    # 正常: 1、異常: -1
    #散布図をプロットして外れ値を確認
    plt.figure()
    ax = plt.subplot()
    # 正常データのプロット
    ax.scatter(np.where(pred > 0), array_df_filtered[np.where(pred > 0)])
    # 異常データのプロット
    ax.scatter(np.where(pred < 0), array_df_filtered[np.where(pred < 0)])
    savefig(prms, "lofscat")    

    plt.figure()
    ax = plt.subplot()
    # 正常データのプロット
    ax.scatter(array_df_filtered[np.where(pred > 0)], array_df_filtered[np.where(pred > 0)])
    # 異常データのプロット
    ax.scatter(array_df_filtered[np.where(pred < 0)], array_df_filtered[np.where(pred < 0)])
    savefig(prms, "lofscatline")    

    # 正常データだけ
    data_mod = array_df_filtered[np.where(pred > 0)]
    if len(data_mod) == 0:
        brainresult["error"] = "NoFilteredData"
        return False

    date_mod = array_date[np.where(pred > 0)]
    data_mod_comb=np.concatenate([date_mod,data_mod],1)
    df_mod=pd.DataFrame(data_mod_comb, columns=["record_date","y"])
    print(df_mod)
    #データをプロット
    ##df_mod.set_index("record_date", inplace=True)
    plt.figure()
    plt.subplot()
    df_mod["y"].plot()
    savefig(prms, "lof")    

    plthistgraph(prms, "lofhist", df_mod["y"])

    #1次階差
    df_mod_diff=df_mod["y"].diff().dropna()

    plt.figure()
    plt.subplot()
    df_mod_diff.plot()
    savefig(prms, "kaisa")

    #ヒストグラムを確認
    plthistgraph(prms, "kaisahist", df_mod_diff)

    #移動平均のプロット
    #期間は固定だが、将来的にはデータ数に対し変動させる

    #24x7=168hr : 正味運転時間　1週間
    mv1=movingdays1 * 24    # 168
    #24x30=720hr : 正味運転時間 1か月
    mv2=movingdays2 * 24    #720
    #24*90=2160hr : 正味運転時間 3か月
    mv3=movingdays3 * 24    #2160

    plt.figure()
    plt.subplot()
    plt.plot(df_mod['y'], label="Acc")  
    #plt.title("Original")
    savefig(prms, "movingorg")

    minpval = 3

    plt.figure(figsize=(15, 5))
    plt.subplot()
    plt.plot(df_mod["y"].rolling(mv1, min_periods=minpval).mean())
    #plt.title("Moving Average {}".format(mv1))
    plt.tight_layout() 
    savefig(prms, "moving1")

    plt.figure(figsize=(15, 5))
    plt.subplot()
    plt.plot(df_mod["y"].rolling(mv2, min_periods=minpval).mean())
    #plt.title("Moving Average {}".format(mv2))
    plt.tight_layout() 
    savefig(prms, "moving2")

    plt.figure(figsize=(15, 5))
    plt.subplot()
    plt.plot(df_mod["y"].rolling(mv3, min_periods=minpval).mean())
    #plt.title("Moving Average {}".format(mv3))
    plt.tight_layout() 
    savefig(prms, "moving3")

    #np.polyfit(通常最小二乗法によるカーブフィット)*************************************************
    #原系列（事前処理後）に対し
    #全期間対象
    limit1=tholdlow #しきい置1
    limit2=tholdhigh #しきい置2
    if (limit1 == None): limit1 = 0
    if (limit2 == None): limit2 = 0
    
    # 説明変数x
    x=np.arange(len(date_mod))
    # 目的変数y
    y=df_mod["y"].astype('float')

    #近似式の係数
    res1=np.polyfit(x, y, 1)
    res2=np.polyfit(x, y, 2)
    res3=np.polyfit(x, y, 3)

    #近似式の計算
    y1 = np.poly1d(res1)(x) #1次
    y2 = np.poly1d(res2)(x) #2次
    y3 = np.poly1d(res3)(x) #3次
    #グラフ表示
    plt.figure(figsize=(15, 5))
    plt.plot(x, y, label='org')
    plt.plot(x, y1, label='1dim')
    plt.plot(x, y2, label='2dim')
    plt.plot(x, y3, label='3dim')
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r",label='limit')
    plt.tight_layout() 
    plt.legend()
    savefig(prms, "lsa")

    print("1次 : {}".format(res1))
    print("2次 : {}".format(res2))
    print("3次 : {}".format(res3))

    #寿命計算
    #とりあえず1次のみ

    eq1 = np.poly1d(res1)

    if (eq1.coef[0] > 0):
        #定数bをlimitまで下げる
        b1 = eq1.coef[1] - limit1
        b2 = eq1.coef[1] - limit2
        # y = 0として解く
        # y= ax + b
        # 0 = ax + b
        # x = -b /a
        jyum1 = b1 * -1 / eq1.coef[0]
        jyum2 = b2 * -1 / eq1.coef[0]
        brainresult["output"]["lifetime1"] = "{:,}".format(int(jyum1))
        brainresult["output"]["lifetime2"] = "{:,}".format(int(jyum2))
        print("寿命1", jyum1)
        print("寿命2", jyum2)
    else:
        print("係数がマイナス寿命測定できません")
    
    return True

    #limit1
    poly_future1=len(x)
    poly_pred1=np.poly1d(res1)(poly_future1) #1次のみ

    #limit2
    poly_future2=len(x)
    poly_pred2=np.poly1d(res1)(poly_future2) #1次のみ
    return True

    if poly_pred1 > limit1:
            print("既にしきい値1を超えています")  

    while poly_pred1 < limit1:
        poly_future1+=1
        poly_pred1=np.poly1d(res1)(poly_future1)
        print(poly_pred1,limit1)
        if poly_pred1 >= limit1:
            print("注意しきい値まで：{} hr - {} days".format(poly_future1-len(x), (poly_future1-len(x))//24))
            break
            
    if poly_pred2 > limit2:
            print("既にしきい値2を超えています")  

    while poly_pred2 < limit2:
        poly_future2+=1
        poly_pred2=np.poly1d(res1)(poly_future2)
        
        if poly_pred2 >= limit2:
            print("危険しきい値まで：{} hr - {} days".format(poly_future2-len(x), (poly_future2-len(x))//24))
            break

    return True

#メイン処理
def main():
    args = sys.argv
    params = trcom.params(args)

    if (params.guid == "braintest01"):
        trend_plot(params)
    else:
        try:
            if trend_plot(params):
                brainresult["status"] = True
            else:
                brainresult["status"] = False
        except Exception as e:
            brainresult["error"] = str(e)
            brainresult["errordetail"] = traceback.format_exc()
            print(e)

    params.save_result(brainresult)

if __name__=='__main__':
    main()
