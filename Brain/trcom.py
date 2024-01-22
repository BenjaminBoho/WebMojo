# モジュールのインポート
import json
#import mysql.connector
import pandas as pd
import datetime
import pymysql.cursors

#*************************************************************
# mysql.connector はサーバー開発機ではストアドプロシジャの実行ができなかった multi=Trueにしないさいエラー
# 代わりに pymysqlを使用する
#*************************************************************
#from attrdict import AttrDict

global dbserver

def font_setup():
    from matplotlib import rcParams
    # rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    rcParams['font.sans-serif'] = ['meiryo']
    return

class params:
    params = {}
    workfolder = ""
    guid = ""
    def __init__( self, args) :
        if len(args) < 2:
            self.guid = "braintest01"
            self.workfolder = "./" + self.guid
        else:
            self.guid = args[1]
            self.workfolder ='../work/' + self.guid 
        # configファイルを開く
        json_file = open(self.workfolder + '/params.json', 'r',  encoding='utf-8')
        # JSONとして読み込む
        #params =  AttrDict(json.load(json_file))
        params = dict2(json.load(json_file))
        params["guid"] = self.guid
        print (params)
        self.params = params
        global dbserver
        dbserver = params["dbserver"]
        return
        
    def save_result(self, result):
        fw = open(self.workfolder + '/result.json','w')
        json.dump(result,fw,indent=4)


class dict2(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self         

# トレンドデータを取得 仮想履歴対応
def get_trend(sensortype, sensorid, fm, to):
    stmt = "CALL tcom.`V_trends`(%s, %s ,%s, %s,false,false,true,true,0, false, 0);"
    prms = (sensortype, sensorid, fm, to,)
    return pd.read_sql_query(stmt, dbcon(), params=prms)
    #return dbget(stmt, prms)

# fftデータを取得 仮想履歴対応
def get_fft(sensortype, sensorid, fm, to):
    stmt = "CALL tcom.`V_fft`(%s, %s ,%s, %s,'',null,false,true,false,false);"
    prms = (sensortype, sensorid, fm, to,)
    return pd.read_sql_query(stmt, dbcon(), params=prms)
    #return dbget(stmt, prms)

# 物理センサID、日付指定でFFTデータを取得する
def get_fft_p(sensortype, sensorid, dt):
    stmt = "select * from fftshort where sensortype =%s and sensorid =%s and fftrecord_date =%s"
    prms = (sensortype, sensorid, dt)
    return pd.read_sql_query(stmt, dbcon(), params=prms)

# adver_log からアドバタイズのログを取得する
def get_adver(sensortype, sensorid, dt):
    stmt = "select recordDate,max(Battery) as Battery , max(VRMSXAcc) as VRMSXAcc from adver_log where sensortype =%s and sensorid =%s and recordDate >= %s group by SensorType ,SensorId ,recordDate;"
    prms = (sensortype, sensorid, dt)
    return pd.read_sql_query(stmt, dbcon(), params=prms)

# DBコネクション取得
def dbcon():
    try:
        # configファイルを開く
        json_file = open('config.json', 'r')
        # JSONとして読み込む
        config  = json.load(json_file)
        sqlcon = config["sqlconnection"]
        if not(dbserver is None or dbserver == ""):
            sqlcon["host"] = dbserver
        cnx = pymysql.connect(**sqlcon)
        #cnx = mysql.connector.connect(**sqlcon)
        return cnx

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
        return None

# DBからステートメントを実行して結果を取得する
def dbget(stmt, prms):
    results = []
    try:
        cnx = dbcon()
        cursor = cnx.cursor()

        cursor.execute(stmt, prms)
        rows = cursor.fetchall()
        for row in rows:
            x = dict(zip([d[0] for d in cursor.description], row))
            results.append(x)
        cursor.close()
        cnx.close()

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
        return None

    return results

# 単位変換ベースクラス
class UnitConvBase:
    unitinfo = {}   #継承先で定義する
    def conv(self, fmunit, tounit, val):
        return val * self.get_coef(fmunit, tounit)
    def get_coef(self, fmunit, tounit):
        return self.unitinfo[fmunit]["coef"] / self.unitinfo[tounit]["coef"]
    def unitname(self, unit):
        return self.unitinfo[unit]["name"]
# 単位変換 速度
class speedclass(UnitConvBase):
    unitinfo = {
        "M":{"coef":1, "name":"m/s"},
        "MM":{"coef":0.001, "name":"mm/s"},
        "I":{"coef":0.0254, "name":"ips"}
    }    
    def __init__( self) :
        print("speedclass")

# 単位変換 加速度
class accelerationclass(UnitConvBase):
    unitinfo = {
        "M":{"coef":1, "name":"m/s²"},
        "MM":{"coef":0.001, "name":"mm/s²"},
        "I":{"coef":0.0254, "name":"ips²"},
        "G":{"coef":9.80665, "name":"G"}
    }  
    def __init__( self) :
        print("accelerationclass")

# 単位変換呼び出し
class UnitConv:
    speed = speedclass()
    acceleration = accelerationclass()

