import matplotlib.pyplot as plt
import numpy as np
import trcom

def fft_plot():
     results = trcom.get_fft("M", "VB3909B5", "2019-12-16 23:59:36", None)
     fftdata =[]
     for rec in results:
          fftdata = rec["fftdatax"].split(",")
     print ("end")     

     # figureの初期化
     fig = plt.figure()
     ax1 = fig.add_subplot(111)
     # データのプロット
     y =  [float(s) for s in fftdata]
     x = np.linspace(1,10000, 10000)
     ax1.plot(x, y)

     #plt.tight_layout()
     plt.show()

     # figureの保存
     #plt.savefig("foo.png")
     return

def trend_plot():
     x = []
     y = []
     results = trcom.get_trend("M", "VB3909B5", "2019-11-16 23:59:36", "2020-1-22 23:00:00")
     for rec in results:
          x.append(rec["record_date"])
          y.append(float(rec["x_vrmsacc"]))
          print(rec)
     print ("end")     
     # figureの初期化
     fig = plt.figure()
     ax1 = fig.add_subplot(111)
     # データのプロット
     ax1.plot(x, y)
     plt.show()

     # figureの保存
     #plt.savefig("foo.png")
     return


#メイン処理
trend_plot()     