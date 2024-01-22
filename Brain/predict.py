import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
import seaborn as sns
import warnings
import openpyxl
import faiss
import json
import traceback

def predict(workpath, pkl_path, query_path, none_image_file):

    query_mv_image_name = "df_query_mv.png"
    result_xqxb_image_name = "result(xq+xb).png"
    result_xb_image_name = "result(xb).png"
    result_json = {}

    df_norm_mv = pd.read_pickle(pkl_path)

    #クエリFFT（判定したいFFT）の読み込み
    #正常と異常を読み込み、差分をとる（差分FFT）
    #差分FFTのマイナス値は0にする。
    #差分FFTをacc_windowで平滑化してクエリとする。
    #対象は加速度FFTのみ。

    #移動平均による平滑化
    acc_window=3 #ハイパラ

    df_query=pd.read_csv(query_path, index_col=0)
    df_query["diff"]=df_query["abnormal"]-df_query["normal"].clip(0,) #差分を計算　負の値は０にする

    #移動平均
    df_query_mv=df_query.rolling(acc_window,axis=0,center=True).mean()
    df_query_mv.fillna(0,inplace=True)

    #クエリを作成
    xq=df_query_mv["diff"].values.copy(order='C').reshape(1,9993).astype("float32")

    d=9993 #加速度FFTの次元
    index=faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT) #インデックスの作成 INNER_PRODUCT

    #教師DB
    xb=df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[:,9:].values.copy(order='C').astype(np.float32)
    faiss.normalize_L2(xb) #cosine similarityの時必要
    index.add(xb)

    faiss.normalize_L2(xq) #cosine similarityの時必要

    #次元数が4の倍数かつ同時検索クエリ数が20件以下の場合はSSEを使って距離計算,
    #それ以上ではBLAS実装に計算処理が委譲される。マルチスレッドで処理される箇所はOpenMPを使って実装されている

    #上位何位まで検索するか？
    k=3

    #クエリFFTでk-近傍探索
    D,I=index.search(xq,k) # actual search

    df_results=pd.concat(
        [pd.DataFrame(I,columns=["I_1st","I_2nd","I_3rd"]),
         pd.DataFrame(D,columns=["D_1st","D_2nd","D_3rd"])],
        axis=1)
    
    #Dist(距離)以上のみを抽出
    Dist=0.5 #ハイパラ
    df_results_pick=df_results[df_results["D_1st"]>Dist]

    #推論結果を保存
    result_json["AILabelList"] = []
    result_json["DList"] = []
    for i in range(k):
        result_json["AILabelList"].append(df_norm_mv[df_norm_mv['status']=='Diff'].iloc[I[0][i],6])
        result_json["DList"].append(f"{D[0][i]:.3f}")

    # 636 
    # 推定結果2位までのD（距離）の合計がD≧0.65であり、
    # 且つAI_labelに同じものが含まれていれば有効（AIラベル自動、メール自動）とする。
    # その際の原因推定結果（AI_label）は共通のものだけとする。

    Dans = []
    Dval = D[0][0] + D[0][1]
    result_json["AILabel"] = ""
    result_json["D"] = f"{Dval:.3}"
    if (Dval) >= 0.65: 
        for lbl in result_json["AILabelList"][0].split('&'):        #第一の結果ラベルごとに
            if lbl in result_json["AILabelList"][1].split('&'): Dans.append(lbl)    #第二位に含まれていれば採用
        result_json["AILabel"] = '&'.join(Dans) # &で連結

    #result_json["AILabel"] = df_norm_mv[df_norm_mv['status']=='Diff'].iloc[I[0][0],6]
    #result_json["D"] = f"{D[0][0]:.3f}"
    
    if none_image_file == False:

        #クエリFFTの表示
        fig=plt.figure(figsize=(15,3))
           
        plt.subplot(1,3,1)
        plt.title("normal")
        plt.plot(df_query_mv["normal"], alpha=1.0, color="b", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_query.max().max()])
        plt.xlabel('Hz')
        plt.ylabel('ACC')
           
        plt.subplot(1,3,2)
        plt.title("abnormal")
        plt.plot(df_query_mv["abnormal"], alpha=1.0, color="r", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_query.max().max()])
           
        plt.subplot(1,3,3)
        plt.title("diff")
        plt.plot(df_query_mv["diff"], alpha=1.0, color="g", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_query.max().max()])
           
        plt.tight_layout()
        fig.savefig(workpath + "/" + query_mv_image_name)
        plt.close()
        result_json["queryImageName"] = query_mv_image_name

        #クエリと原因
        fig=plt.figure(figsize=(10, k*3))

        #加速度FFT
        for i in range (len(xq)):
            plt.subplot(k+1,len(xq),i+1)
            plt.plot(np.arange(8,10001),xq[i], alpha=1.0, color="r", label=None, linestyle="-", linewidth=1.0, marker=None)
            plt.ylim([0,xq.max().max()])
            plt.title("query"+str(i+1))
            plt.xlabel('Hz')
            plt.ylabel('ACC')
    
            for j in range (k):
                plt.subplot(k+1,len(xq),len(xq)+(i+1)*(j+1))
                plt.plot(np.arange(8,10001), df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i][j],9:], alpha=1.0, color="b",
                         label=f"Index={df_norm_mv[df_norm_mv['status']=='Diff'].index[I[i][j]]}, D={D[i][j]:.3f}", linestyle="-", linewidth=1.0, marker=None)
                plt.title(df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i][j],5])
                plt.ylim([0,df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i],9:].max().max()])
                plt.xlabel('Hz')
                plt.ylabel('ACC')
                plt.legend()
            plt.tight_layout()

        fig.savefig(workpath + "/" + result_xqxb_image_name)
        plt.close()
        result_json["resultXqxbImageName"] = result_xqxb_image_name

        #原因のみ
        fig=plt.figure(figsize=(k*5, len(xq)*5))

        for i in range (len(xq)):
    
            for j in range (k):
                plt.subplot(len(xq)+1,k,(i*3)+(j+1))
                plt.plot(np.arange(8,10001), df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i][j],9:], alpha=1.0, color="b",
                         label=f"Index={df_norm_mv[df_norm_mv['status']=='Diff'].index[I[i][j]]}, D={D[i][j]:.3f}", linestyle="-", linewidth=1.0, marker=None)
                plt.title(df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i][j],5])
                plt.ylim([0,df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[I[i],9:].max().max()])
                plt.xlabel('Hz')
                plt.ylabel('ACC')
                plt.legend()
            plt.tight_layout()

        plt.tight_layout()
        fig.savefig(workpath + "/" + result_xb_image_name)
        plt.close()
        result_json["resultXbImageName"] = result_xb_image_name

    return result_json

#メイン処理
def main():

    guid = ""
    workpath = ""
    args = sys.argv
    result_json = {}

    warnings.filterwarnings('ignore')
    sns.set()

    try:
        
        if len(args) < 2:
            guid = 'braintest01'
            workpath = "./" + guid           
        else:
            guid = args[1]
            workpath ='../work/' + guid

        parampath = workpath + '/params.json'
        none_image_file = False
        if os.path.isfile(parampath):
            json_file = open(parampath, 'r',  encoding='utf-8')
            params = dict2(json.load(json_file))
            none_image_file = bool(params["NoneImageFile"])

        query_path = workpath + '/query.csv'
        pkl_path = '../work/CauseEstimation/Teacher.pkl'

        result_json = predict(workpath, pkl_path, query_path, none_image_file)
    except Exception as e:
        result_json["error"] = str(e) + "\n" + traceback.format_exc()
 
    fw = open(workpath + '/result.json','w')
    json.dump(result_json,fw,indent=4) 

class dict2(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self   

if __name__=='__main__':
    main()