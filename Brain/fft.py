import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import numpy as np
import pandas as pd
import sys
import time
import json
import traceback
import heapq as hq
import trcom

brainresult = {"status":False, "files":[]}
trcom.font_setup()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#ファイル名をつけて図を保存
def savefig(prms, graphid):
    fnamefull = graphid + ".png"
    plt.savefig(prms.workfolder + "/" + fnamefull)
    return fnamefull

# FFTデータの単位変換を行ったリストを返す
def unit_convert(df, sensortype, axis, unitsystem):
    dls = []
    if len(df) == 0:
        return dls
    #Y軸データを取得
    itm = ""
    if (sensortype == "M"):
        itm ="fftdatax"
    else :
        if (axis == "X"):
            itm ="fftdatax"
        if (axis == "Y"):
            itm ="fftdatay"
        if (axis == "Z"):
            itm ="fftdataz"
    valitm = np.array(df[itm].str.split(","))
    for i in range(len(df)):
        #カンマ区切りのFFTをarrayにしてリストに追加
        valtimlist = valitm[i]
        if (sensortype == "I"): #iAlertのときは1つ目はinfinityなので削除
            valtimlist.pop(0)
        dls.append(np.array(valtimlist).astype(float))

    #単位変換
    # センサタイプ M は元がm/s2 単位系をEにするときは ips2
    # センサタイプ I は元がips 単位系をMにするときは mm/s
    coef = 1
    if (sensortype == "M" and unitsystem == "E"):
        coef = trcom.UnitConv.acceleration.get_coef("M", "I")
    elif (sensortype == "I" and unitsystem == "M"):
        coef = trcom.UnitConv.speed.get_coef("I", "MM")
    for i in range(len(df)):
        dls[i] = dls[i] * coef    
    return dls

# 卓越周波数のトラッキング
def get_dominant(prms, topdom):
    prm = prms.params
    tracks = []
    #データ取得
    df = trcom.get_fft(prm.sensortype, prm.sensorid, prm.fromdt, prm.todt)
    dl = unit_convert(df, prm.sensortype, prm.axis, prm.unitsystem)
    dts = df["fftrecord_date"]       
    #卓越周波数のトラッキング
    for dom in topdom:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        hzi = dom[1]    #周波数インデックス
        hz = hzi + 1    #周波数 
        #前後1Hzも合わせて取得
        for idx in range(hzi - 1, hzi + 2):
            if idx >= 0:
                samefreq = []
                for d in dl:
                    samefreq.append(d[idx])
                    print(idx, ",", d[idx])
                ax.plot(dts, samefreq, label="{}Hz".format(idx))
        plt.title("{}Hz".format(hz))
        plt.legend()
        fname = "hz{}".format(hz)
        savefig(prms, fname)
        plt.close()
        tracks.append(hz)
    return tracks

#表示単位
def get_dispunit(prms):
    prm = prms.params
    if (prm.sensortype == "M"):
        if  (prm.unitsystem == "E"):
            return trcom.UnitConv.acceleration.unitname("I")
        else:
            return trcom.UnitConv.acceleration.unitname("M")    
    if (prm.sensortype == "I"):
        if  (prm.unitsystem == "E"):
            return trcom.UnitConv.speed.unitname("I")
        else:
            return trcom.UnitConv.speed.unitname("MM")
    return ""

#FFTデータ
def fft_plot(prms):
    prm = prms.params
    
    #データの単位
    dispunit = get_dispunit(prms)

    #卓越周波数のみ
    if prm.dominantonly:
        doms = [(0, prm.hz -1)]
        tracks = get_dominant(prms, doms)
        brainresult["tracking"] = tracks
        return True

    #FFTを取得して単位変換したlistを返す
    df = trcom.get_fft(prm.sensortype, prm.sensorid, prm.tgtdt, None)
    dl = unit_convert(df, prm.sensortype, prm.axis, prm.unitsystem)
    if (len(dl) < 1):
        brainresult["error"] = "DataNotFound"
        return False
    plt.figure()
    plt.subplot()
    plt.plot(dl[0])
    plt.ylabel(dispunit)
    #plt.tight_layout() 
    brainresult["files"].append(savefig(prms, "org"))

    #卓越周波数
    maxdomfreq = 5  #大きいものから選択する数
    domfreqs = []
    topdom = []
    for fidx, f in enumerate(dl[0]):
        if f > 0: domfreqs.append((f, fidx))    #値が0以上
    topdom = sorted(domfreqs, reverse=True)[:maxdomfreq]   #値が大きいもの順に
    tracks = []
    if len(topdom) > 0:
        tracks = get_dominant(prms,topdom)
        brainresult["tracking"] = tracks

    #OK FFT
    for okcnt,okdt in enumerate(prm.OKFFTS):
        df = trcom.get_fft_p(prm.sensortype, okdt["sensorid"], okdt["dt"])
        dl = unit_convert(df, prm.sensortype, prm.axis, prm.unitsystem)
        if (len(dl) < 1):
            brainresult["error"] = "DataNotFound"
            return False
        plt.figure()
        plt.subplot()
        plt.plot(dl[0])
        plt.ylabel(dispunit)
        fname = "ok{}".format(okcnt)
        okdt["file"] = savefig(prms, fname)
    brainresult["OKFFTS"] = prm.OKFFTS
    
    return True

#メイン処理
def main():
    args = sys.argv
    params = trcom.params(args)

    if (params.guid == "braintest01"):
        fft_plot(params)
    else:
        try:
            fft_plot(params)
        except Exception as e:
            brainresult["error"] = str(e)
            brainresult["err5ordetail"] = traceback.format_exc()
            print(e)

    params.save_result(brainresult)

if __name__=='__main__':
    main()