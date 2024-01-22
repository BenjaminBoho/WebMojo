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

def make_model(workpath, teacher_db_path):

    result_json = {}
    fft_image_name = "xb(mv).png"

    #教師DBの読み込み
    #status Normalが正常FFT、Abnormalが異常時FFT、Diffがその差分（異常時-正常）
    df=pd.read_excel(teacher_db_path, index_col=None, sheet_name="faiss_xb",header=0, parse_dates=["date"], engine="openpyxl")
    df.dropna(subset=["no","machine"], how='all', inplace=True) #欠損行を削除

    #データを正規化
    df_norm=df.copy()

    #Diffの負値を０に置き換え
    df_norm=pd.concat([df_norm.iloc[:,:9],df_norm.iloc[:,9:].clip(0,)],axis=1)

    #移動平均による平滑化
    acc_window=3 #ハイパラ

    df_norm_mv=df_norm.iloc[:,9:].rolling(acc_window,axis=1,center=True).mean()
    df_norm_mv.fillna(0,inplace=True)
    df_norm_mv=pd.concat([df.iloc[:,:9],df_norm_mv],axis=1)
    df_norm_mv

    #教師FFTの表示
    n_of_fft=len(df_norm)//3

    fig=plt.figure(figsize=(15, n_of_fft*3))

    #正常FFT
    for i in range (n_of_fft):
        plt.subplot(n_of_fft,3,3*i+1)
        plt.title(df_norm_mv[df_norm_mv["status"]=="Normal"].iloc[i,4])
        plt.plot(np.arange(8,10001), df_norm_mv[df_norm_mv["status"]=="Normal"].T.iloc[9:,i], alpha=1.0, color="b", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_norm_mv.iloc[i*3:i*3+3,9:].max().max()])
        plt.xlabel('Hz')
        plt.ylabel('ACC')

    #異常FFT
    for i in range (n_of_fft):
        plt.subplot(n_of_fft,3,3*i+2)
        plt.title(df_norm_mv[df_norm_mv["status"]=="Abnormal"].iloc[i,5])
        plt.plot(np.arange(8,10001), df_norm_mv[df_norm_mv["status"]=="Abnormal"].T.iloc[9:,i], alpha=1.0, color="r", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_norm_mv.iloc[i*3:i*3+3,9:].max().max()])
        plt.xlabel('Hz')
        plt.ylabel('ACC')

    #差分FFT
    for i in range (n_of_fft):
        plt.subplot(n_of_fft,3,3*i+3)
        plt.title(df_norm_mv[df_norm_mv["status"]=="Diff"].iloc[i,4])
        plt.plot(np.arange(8,10001), df_norm_mv[df_norm_mv["status"]=="Diff"].T.iloc[9:,i], alpha=1.0, color="g", label=None, linestyle="-", linewidth=1.0, marker=None)
        plt.ylim([0,df_norm_mv.iloc[i*3:i*3+3,9:].max().max()])
        plt.xlabel('Hz')
        plt.ylabel('ACC')

    plt.tight_layout() 

    fig.savefig(workpath + "/" + fft_image_name)

    #教師DB
    dirpath = '../work/CauseEstimation'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    df_norm_mv.to_pickle(dirpath + '/Teacher.pkl')
    result_json["imageName"] = fft_image_name

    return result_json

#メイン処理
def main():

    guid = ""
    args = sys.argv
    result_json = {}

    warnings.filterwarnings('ignore')
    sns.set()

    try:
        if len(args) < 2:
            guid = "braintest01"
            workpath = "./" + guid
        else:
            guid = args[1]
            workpath ='../work/' + guid
        teacher_db_path ='../work/CauseEstimation/TeacherDB.xlsx' 

        result_json = make_model(workpath, teacher_db_path)
    except Exception as e:
        result_json["error"] = str(e) + "\n" + traceback.format_exc()
 
    fw = open(workpath + '/result.json','w')
    json.dump(result_json,fw,indent=4)
    fw.close()

if __name__=='__main__':
    main()