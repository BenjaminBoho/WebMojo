import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import numpy as np
import pandas as pd
import sys
import time
import json
import traceback
import datetime

import trcom
import daily
import suggest_threshold
import rising_trend_alarm

import seaborn as sns
sns.set()

minvaltacc = 1.0 #加速度1.0m/s2
minvalspeed = 0.3 #速度0.3mm/s

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#ファイル名をつけて図を保存
def savefig(prms, graphid):
    noimagefile = prms.params["noimagefile"]
    if noimagefile: return
    plt.savefig(prms.workfolder + "/" + graphid + ".png")
    prms.params["brainresult"]["files"].append(graphid + ".png")
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
    plt.title(graphid,fontdict={'fontsize': 30})
    ax.set_ylabel(dispunit)
    #plt.title("日本語")
    savefig(prms, graphid)
    return

#シリーズのヒストグラム
def plthistgraph(prms, graphid, ds):
    plt.figure()
    ax = plt.subplot()
    ax.hist(ds, bins=30)
    plt.title(graphid,fontdict={'fontsize': 30})
    savefig(prms, graphid)
    return

#トレンドデータ
def trend_plot(prms):

    prm = prms.params
    brainresult = prm["brainresult"]
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
    timezoneoffsetprm = prm["timezoneoffset"]
    judgedays = prm["judgedays"]
    verylonghours = prm["verylonghours"]
    noimagefile = prm["noimagefile"]
    cluster_number = prm["cluster_number"]
    sigma = prm["sigma"]
    alarm_limit = prm["alarm_limit"]
    nolifespancalcdays = prm["nolifespancalcdays"]
    exectrendrisingonly = prm["exectrendrisingonly"]

    #タイムゾーンオフセットをtimedeltaフォーマットにしておく
    timezoneoffset = datetime.timedelta(hours=timezoneoffsetprm)
    todttime = datetime.datetime.strptime(todt, '%Y/%m/%d %H:%M:%S')

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
        df = results.loc[:,["record_date", "x_vrmsacc"]]
    else :
        if (axis == "X"):
            df = results.loc[:,["record_date", "x_vrms"]]
        if (axis == "Y"):
            df = results[:,["record_date", "y_vrms"]]
        if (axis == "Z"):
            df = results[:,["record_date", "z_vrms"]]

    #Y軸名を統一
    df.columns = ["record_date", "y"]

    #欠損値行を削除
    df.dropna(how="any",inplace=True)

    if (len(df) == 0):
        brainresult["error"] = "DataNotFound"
        return False

    #UTCからローカル時刻への変換
    df.loc[:,("record_date")]=df.loc[:,("record_date")] + timezoneoffset

    #単位変換
    # センサタイプ M は元がips2 単位系をMにするときは m/s2
    # センサタイプ I は元がips 単位系をMにするときは mm/s
    if (sensortype == "M" and unitsystem == "M"):
        coef = trcom.UnitConv.acceleration.get_coef("I", "M")
        df.loc[:,"y"] = df.loc[:,"y"] * coef
    elif (sensortype == "I" and unitsystem == "M"):
        coef2 = trcom.UnitConv.speed.get_coef("I", "MM")
        df.loc[:,"y"] = df.loc[:,"y"] * coef2

    #indexを時間列に変更
    df.set_index(["record_date"], inplace=True)

    # トレンド上昇アラームしきい値の計算
    rising_trend_alarm.rising_trend_alarm(df, prms)

    # トレンド上昇アラームしきい値の計算のみを実施したいときはここで処理終わり
    if (exectrendrisingonly):
        return
    
    # しきい値の自動提案
    # 廃止 suggest_threshold.suggest_threshold(df, prms)

    nolifeday = todttime + timezoneoffset - datetime.timedelta(days=nolifespancalcdays) #比較する日付
    #lastnewdt = df.loc[:,("record_date")].max() -timezoneoffset
    lastnewdt = df.index.max() -timezoneoffset
    # 最新のトレンドデータ日付 寿命計算異常判断用
    brainresult["output"]["lasttrenddate"] = lastnewdt.strftime('%Y-%m-%d %H:%M:%S')
    # 最新データが寿命計算の有効日付より古い場合は処理をスキップ
    if (nolifeday > lastnewdt):
        brainresult["output"]["nonewdata"] = True
        return
    
    # 原系列
    pltdategraph(prms, "org", df, dispunit)

    # 原系列ヒストグラム
    plthistgraph(prms, "orghist" ,df["y"])

    #しきい値によるフィルター
    df_filtered= df[df["y"] > minval]
    if (maxval != None):
        df_filtered = df[df["y"] < maxval]
    if len(df_filtered) < judgedays * 24:
        brainresult["error"] = "NoFilteredData"
        return False

    pltdategraph(prms, "filtered", df_filtered, dispunit)
    plthistgraph(prms, "filtered_hist" , df_filtered["y"])
  
    #LOFによるトレンド急変
    
    #LOF前にトレンド除去の為、1次階差をとる
    #df_filtered_diff=df_filtered["y"].diff().dropna()
    
    #移動平均との差をとる_2021.9
    #データ数ではなく、日数で判断
    movingdays0=3
    mv0=str(movingdays0) + "D"
    
    minpval2 = movingdays0*2 #最低必要データ数
    
    plt.figure(figsize=(10, 4.8))
    plt.subplot()
    plt.plot(df_filtered["y"], label="Acc")  
    plt.plot(df_filtered["y"].rolling(mv0, min_periods=minpval2).mean())
    plt.title("Moving Average {}".format(mv0),fontdict={'fontsize': 30})
    plt.tight_layout() 
    savefig(prms,"Moving Average {}".format(mv0))
    
    df_filtered_mv=pd.DataFrame(df_filtered["y"].rolling(mv0,min_periods=minpval2).mean())
    df_filtered_diff=(df_filtered-df_filtered_mv).dropna()

    pltdategraph(prms, "filtered_diff", df_filtered_diff, dispunit)
    plthistgraph(prms, "filtered_diff_hist" , df_filtered_diff["y"])

    #arrayに変換
    array_df_filtered_diff= np.array(df_filtered_diff.values.reshape(-1, 1))
    array_date_diff = np.array(df_filtered_diff.index.to_pydatetime().reshape(-1, 1))
    
    #階差なしのデータをarrayに変換（後々必要）
    array_df_filtered=df_filtered.values.reshape(-1, 1)
    array_date=df_filtered.index.to_pydatetime().reshape(-1, 1)
    
    #LOF
    #クラスター数をcluster_numberとし、各クラスターからの外れ度で異常検知
    #まずはknnから
    
    #近傍データの数
    neinum = len(array_df_filtered)//cluster_number #knnでいうk
    if (neinum <= 0):
        brainresult["error"] = "NotEnoughDataToTarget"
        return False
    
    def k_nearest_neighbors(pt, pts):
        # 各点におけるその他全ての点までの距離 (N x N)
        distance_to_every_point = np.linalg.norm(pt - pts, axis=1)
    
        # 上記を昇順にソートし、その元インデックスを取得
        sorted_indices = np.argsort(distance_to_every_point)

        # 上記元インデックスから元データを昇順に並び替え（KNN距離の小さい順に元データを並び替え）
        sorted_pts = pts[sorted_indices] 
    
        # 最初のデータ自分自身なので削除(KNN距離は0)
        sorted_pts = sorted_pts[1:]
    
        # 上位k個を取り出し
        return sorted_pts[:neinum] # neinum=kの事
    
    KNN_table = [] #上位k個のデータをまとめる

    for i in range(array_df_filtered_diff.shape[0]):
        KNN_table.append(k_nearest_neighbors(array_df_filtered_diff[i], array_df_filtered_diff))
    
    #各点に対応する上位k個のデータ
    knn_lookup_table = {}
    for i in range(array_df_filtered_diff.shape[0]):
        knn_lookup_table[tuple(array_df_filtered_diff[i])] = KNN_table[i]
    
    def reachability_distance(pt1, pt2):
        true_distance = np.linalg.norm(pt1 - pt2) 
        other_knn_distance = np.linalg.norm(pt2 - knn_lookup_table[tuple(pt2)][-1]) #距離
        return max([true_distance, other_knn_distance]) #大きい方をとる
    
    local_reach_density_lookup_table = {}
    for i in range(array_df_filtered_diff.shape[0]):
        pt = array_df_filtered_diff[i]
        knns = KNN_table[i] #上位k個のデータ
        sum_reachability = 0
        for j in range(len(knns)):
            sum_reachability = sum_reachability + reachability_distance(pt, knns[j])
        avg_reachability = sum_reachability / neinum
        local_reach_density_lookup_table[tuple(pt)] = (1/avg_reachability) #各点の局所密度
    
    LOFs = []
    Local_Outlier_Factor_lookup_table = {}
    #Calculate the LoF for each point
    for i in range(array_df_filtered_diff.shape[0]):
        pt = array_df_filtered_diff[i]
        knns = KNN_table[i] #上位k個のデータ
        lrd_knns = 0
        for j in range(len(knns)):
            lrd_knns = lrd_knns + local_reach_density_lookup_table[tuple(knns[j])]
        avg_lrd = lrd_knns / neinum
        LOFs.append(avg_lrd/local_reach_density_lookup_table[tuple(pt)])
        Local_Outlier_Factor_lookup_table[tuple(pt)] = (avg_lrd/local_reach_density_lookup_table[tuple(pt)]) #各点のLOF  
    #LOFが大きいほど外れ値
    
    #散布図とヒストグラムをプロットして外れ値を確認
    plt.figure()
    ax = plt.subplot()
    ax.scatter(array_df_filtered_diff,LOFs)
    plt.title("lofscat",fontdict={'fontsize': 30})
    savefig(prms, "lofscat")    

    plt.figure()
    ax = plt.hist(np.array(LOFs)[np.isfinite(LOFs)], bins=100)
    plt.title("lofhist",fontdict={'fontsize': 30})
    savefig(prms, "lofhist")  
    
    LOF_df_diff=pd.DataFrame({'Acc_diff':array_df_filtered_diff.reshape(-1),'LOF':LOFs})
    
    #LOF_df_diffのインデックスを振りなおす
    LOF_df_diff=LOF_df_diff.set_index(df_filtered_diff.index)

    #元データと結合
    df_Acc_LOF=pd.concat([df_filtered,LOF_df_diff], axis=1)
    
    #外れ値を抽出
    #LOFのﾋストグラムより
    
    #ハイパーパラメーター
    #sigma=3 : 99.7％のデータは正常とする場合
    LOF_limit=df_Acc_LOF["LOF"].mean()+df_Acc_LOF["LOF"].std()*sigma

    df_Acc_outliers=df_Acc_LOF.query("LOF>@LOF_limit")
    
    plt.figure()
    plt.scatter(df_Acc_LOF["Acc_diff"],df_Acc_LOF["LOF"]) 
    plt.scatter(df_Acc_outliers["Acc_diff"],df_Acc_outliers["LOF"])
    plt.title("lofscat_out",fontdict={'fontsize': 30})
    savefig(prms, "lofscat_out")
    
    #外れ値のうち、Acc_diffが alarm_limit以上の場合だけを異常検知とする 
    df_Acc_outliers_plus=df_Acc_outliers.query("Acc_diff>{}".format(alarm_limit))
    
    #外れ値のプロット
    #どのデータを外れ値として検出できたか？
    plt.figure(figsize=(10, 19.2))
    
    #差分データに対し、どれが異常値か？
    plt.subplot(4,1,1)
    plt.title("vs_diff",fontdict={'fontsize': 30})
    df_Acc_LOF["Acc_diff"].plot()
    df_Acc_outliers["Acc_diff"].plot.line(style=["r+"],markersize=12)

    #差分データ（正）に対し、どれが異常値か？
    plt.subplot(4,1,2)
    plt.title("vs_diff_plus",fontdict={'fontsize': 30})
    df_Acc_LOF["Acc_diff"].plot()
    df_Acc_outliers_plus["Acc_diff"].plot.line(style=["r+"],markersize=12)

    #原系列（フィルター後）に対し、どれが異常値か？
    plt.subplot(4,1,3)
    plt.title("vs_org_filtered",fontdict={'fontsize': 30})
    df_Acc_LOF["y"].plot()
    df_Acc_outliers["y"].plot.line(style=["r+"],markersize=12)
    
    #どれがトレンド急変か？
    plt.subplot(4,1,4)
    plt.title("final_results",fontdict={'fontsize': 30})
    df_Acc_LOF["y"].plot()
    df_Acc_outliers_plus["y"].plot.line(style=["r+"],markersize=12)
    
    plt.tight_layout() 
    savefig(prms, "LOF_visualized")  
    
    #LOFにて正常と判断されたデータのみでデータフレーム作成
    #df_Acc_outliersのデータを削除

    #差分が負の場合も異常値なので取り除く
    df_mod=df_Acc_LOF.drop(df_Acc_outliers.index, axis=0)
    
    #外れ値を除いてプロット
    pltdategraph(prms, "df_mod_trend", df_mod["y"], dispunit)
    plthistgraph(prms, "df_mod_hist" , df_mod["y"])  

    #異常検知日時
    print(df_Acc_outliers_plus)
    jday = todttime + timezoneoffset - datetime.timedelta(days=judgedays) #比較する日付
    outliersdays = df_Acc_outliers_plus.index[df_Acc_outliers_plus.index > jday]            #比較する日付より最近のものだけ抽出
    brainresult["output"]["outliers"] = (outliersdays - timezoneoffset).strftime('%Y-%m-%d %H:%M:%S').tolist()
   
    #移動平均
    
    """
    #期間は固定だが、将来的にはデータ数に対し変動させる

    #24x7=168hr : 正味運転時間　1週間
    mv1=movingdays1 * 24    # 168
    #24x30=720hr : 正味運転時間 1か月
    mv2=movingdays2 * 24    #720
    #24*90=2160hr : 正味運転時間 3か月
    mv3=movingdays3 * 24    #2160
    """
    
    #データ数ではなく、日数で判断
    #7日移動平均
    mv1=str(movingdays1) + "D"
    #1か月移動平均
    mv2=str(movingdays2) + "D"
    #3か月移動平均
    mv3=str(movingdays3) + "D"

    minpval = 14 #最低14データが必要 (2 x 7日 = 14)

    plt.figure(figsize=(10, 14.4))
    
    plt.subplot(3,1,1)
    plt.plot(df_mod["y"].rolling(mv1, min_periods=minpval).mean())
    plt.title("Moving Average {}".format(mv1),fontdict={'fontsize': 30})

    plt.subplot(3,1,2)
    plt.plot(df_mod["y"].rolling(mv2, min_periods=minpval).mean())
    plt.title("Moving Average {}".format(mv2),fontdict={'fontsize': 30})

    plt.subplot(3,1,3)
    plt.plot(df_mod["y"].rolling(mv3, min_periods=minpval).mean())
    plt.title("Moving Average {}".format(mv3),fontdict={'fontsize': 30})
    
    plt.tight_layout() 
    savefig(prms, "moving")

    #np.polyfit(通常最小二乗法によるカーブフィット)*************************************************
    #原系列（事前処理後）に対し
    #全期間対象
    limit1=tholdlow #しきい置1
    limit2=tholdhigh #しきい置2
    if (limit1 == None): limit1 = 0
    if (limit2 == None): limit2 = 0
    
    # 説明変数x
    x=np.arange(len(df_mod))
    # 目的変数y
    y=df_mod["y"].astype('float')

    #近似式が計算できるかをチェック
    if (len(x) < 2):
        brainresult["error"] = "ApproximationFormulaCannotBeComputedBecauseThereAreFewElements"
        brainresult["errordetail"] = "要素数 {}".format(len(x))
        return False

    #近似式の係数
    res1=np.polyfit(x, y, 1)
    res2=np.polyfit(x, y, 2)
    res3=np.polyfit(x, y, 3)

    #近似式の計算
    y1 = np.poly1d(res1)(x) #1次
    y2 = np.poly1d(res2)(x) #2次
    y3 = np.poly1d(res3)(x) #3次
    #グラフ表示
    plt.figure(figsize=(14.4, 4.8))
    plt.plot(x, y, label='org')
    plt.plot(x, y1, label='1dim')
    plt.plot(x, y2, label='2dim')
    plt.plot(x, y3, label='3dim')
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r",label='limit')
    plt.tight_layout() 
    plt.legend()
    savefig(prms, "linearRegression")

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
        
        #残り時間として
        jyum1 = b1 * -1 / eq1.coef[0]-len(x)
        jyum2 = b2 * -1 / eq1.coef[0]-len(x)
        
        #jyum1 = b1 * -1 / eq1.coef[0]
        #jyum2 = b2 * -1 / eq1.coef[0] 
        
        # brainresult["output"]["lifetime1"] = int(jyum1)
        # brainresult["output"]["lifetime2"] = int(jyum2)
        print("org - 注意しきい値まで：{:,} hr - {} days".format(jyum1, jyum1//24))
        print("org - 危険しきい値まで：{:,} hr - {} days".format(jyum2, jyum2//24))
    else:
        print("係数がマイナス寿命測定できません")
    
    #np.polyfit(通常最小二乗法によるカーブフィット)による
    #各移動平均期間分の最新データに対し
    
    #目的変数
    #個数ではなく日数指定でデータ取り出し
    mv1_curve=df_mod["y"].rolling(mv1,min_periods=minpval).mean()
    mv1_curve_term=mv1_curve[mv1_curve.index.max()-datetime.timedelta(days=movingdays1):mv1_curve.index.max()]

    mv2_curve=df_mod["y"].rolling(mv2,min_periods=minpval).mean()
    mv2_curve_term=mv1_curve[mv2_curve.index.max()-datetime.timedelta(days=movingdays2):mv2_curve.index.max()]

    mv3_curve=df_mod["y"].rolling(mv3,min_periods=minpval).mean()
    mv3_curve_term=mv1_curve[mv3_curve.index.max()-datetime.timedelta(days=movingdays3):mv3_curve.index.max()]


    #説明変数
    #日付をそのまま計算に使う。
    x_mv1=mv1_curve_term.index.map(pd.Timestamp.timestamp)
    x_mv2=mv2_curve_term.index.map(pd.Timestamp.timestamp)
    x_mv3=mv3_curve_term.index.map(pd.Timestamp.timestamp)

    #近似式の係数
    res_mv1=np.polyfit(x_mv1, mv1_curve_term, 1) #1次
    res_mv2=np.polyfit(x_mv2, mv2_curve_term, 1) #1次
    res_mv3=np.polyfit(x_mv3, mv3_curve_term, 1) #1次

    #近似式の計算
    y_mv1 = np.poly1d(res_mv1)(x_mv1) #mv1
    y_mv2 = np.poly1d(res_mv2)(x_mv2) #mv2
    y_mv3 = np.poly1d(res_mv3)(x_mv3) #mv3

    #グラフ表示
    plt.figure(figsize=(10, 19.2))

    plt.subplot(4,1,1)
    plt.plot(np.arange(len(x) - len(x_mv1),len(x)),mv1_curve_term, color="b")
    plt.plot(np.arange(len(x) - len(x_mv1),len(x)), y_mv1, color="y")
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r")
    plt.title("Mooving Average {} (Total range)".format(mv1),fontdict={'fontsize': 30})

    plt.subplot(4,1,2)
    plt.plot(np.arange(len(x) - len(x_mv2),len(x)), mv2_curve_term, color="b")
    plt.plot(np.arange(len(x) - len(x_mv2),len(x)), y_mv2, color="y")
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r")
    plt.title("Mooving Average {} (Total range) ".format(mv2),fontdict={'fontsize': 30})

    plt.subplot(4,1,3)
    plt.plot(np.arange(len(x) - len(x_mv3),len(x)), mv3_curve_term, color="b")
    plt.plot(np.arange(len(x) - len(x_mv3),len(x)), y_mv3, color="y")
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r")
    plt.title("Mooving Average {} (Total range) ".format(mv3),fontdict={'fontsize': 30})

    plt.subplot(4,1,4)
    plt.plot(x,y)
    plt.plot(np.arange(len(x) - len(x_mv1),len(x)), y_mv1)
    plt.plot(np.arange(len(x) - len(x_mv2),len(x)), y_mv2)
    plt.plot(np.arange(len(x) - len(x_mv3),len(x)), y_mv3)
    plt.hlines([limit1, limit2], 0, len(x), linestyle="dashed", color="r")
    plt.title("Comparison (Total range) ",fontdict={'fontsize': 30})

    plt.tight_layout() 

    print("mv1 : {}".format(res_mv1))
    print("mv2 : {}".format(res_mv2))
    print("mv3 : {}".format(res_mv3))
    
    savefig(prms, "linearRegression_mv")
    
    #寿命計算
    #移動平均から
    def lifetime(res_mv, mvlabel):
        eq_mv = np.poly1d(res_mv)
        if (eq_mv.coef[0] > 0):
        #if not np.isnan(eq_mv.coef[0]): #テスト用 係数がマイナスでも計算
            #定数bをlimitまで下げる
            b1_mv = eq_mv.coef[1] - limit1
            b2_mv = eq_mv.coef[1] - limit2
                
            jyum1_mv = b1_mv * -1 / eq_mv.coef[0]-len(x)
            jyum2_mv = b2_mv * -1 / eq_mv.coef[0]-len(x)

            int_jyum1_mv = int(jyum1_mv)
            int_jyum2_mv = int(jyum2_mv)

            # Pythonのintの最大値はC#で扱えないかも知れないので指定最大値以上にならないようにする
            if int_jyum1_mv > verylonghours: int_jyum1_mv = verylonghours
            if int_jyum2_mv > verylonghours: int_jyum2_mv = verylonghours

            brainresult["output"]["lifetime1"][mvlabel] = int_jyum1_mv
            brainresult["output"]["lifetime2"][mvlabel] = int_jyum2_mv
            
            print("{} - 注意しきい値まで：{:,} hr - {} days".format(mvlabel, jyum1_mv, jyum1_mv//24))
            print("{} - 危険しきい値まで：{:,} hr - {} days".format(mvlabel, jyum2_mv, jyum2_mv//24))
        else:
            brainresult["output"]["lifetime1"][mvlabel] = verylonghours
            brainresult["output"]["lifetime2"][mvlabel] = verylonghours
            print("{} - 係数がマイナス寿命測定できません".format(mvlabel))

    lifetime(res_mv1, "mv1")
    lifetime(res_mv2, "mv2")
    lifetime(res_mv3, "mv3")

    #最もクリティカルな時間 (verylonghours が結果の時は答え出てない)
    brainresult["output"]["lifetime1min"] = min(brainresult["output"]["lifetime1"].values())
    brainresult["output"]["lifetime2min"] = min(brainresult["output"]["lifetime2"].values())
    brainresult["output"]["lifetime1max"] = max(brainresult["output"]["lifetime1"].values())
    brainresult["output"]["lifetime2max"] = max(brainresult["output"]["lifetime2"].values())
    print("クリティカル - 注意しきい値まで：{:,} hr - {} days".format(brainresult["output"]["lifetime1min"], brainresult["output"]["lifetime1min"]//24))
    print("クリティカル - 危険しきい値まで：{:,} hr - {} days".format(brainresult["output"]["lifetime2min"], brainresult["output"]["lifetime2min"]//24))

    return
    
    #mv1からmv1までの寿命計算のうち、最もクリティカルなものを選ぶ
    #危険しきい値までの予寿命が3ヶ月を切ったらメールで通知
    
    #*************************************************************野呂：寿命計算

   

#メイン処理
def main():
    params = daily.initcommon()

    if (params.guid == "braintest01"):
        trend_plot(params)

    params.save_result(params.params["brainresult"])
    return

if __name__=='__main__':
    main()
