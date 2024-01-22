#!/usr/bin/python3
# -*- Coding: utf-8 -*-

# Log設定
import logging
import logging.config
# def setup_logger(name, congfile='Log.conf'):
#     # print('setup_logger awsproc: name=%s' % name)
#     if name=='__main__':
#         logging.config.fileConfig(congfile)
#         logging.getLogger('TRCOM')  # 大元 (複数のプロセス利用を考慮)
#     log = logging.getLogger("TRCOM").getChild(name) # サブプロセス
#     return log
# # Log設定呼び出し   
# # importとの順番によりLogが出たりでなかったりするので注意
# logger = setup_logger(__name__)

import logging
logger = logging.getLogger(__name__)
# fmt = "%(asctime)s %(levelname)s %(name)s %(funcName)s:%(message)s"
fmt = "%(asctime)s:%(funcName)s:%(message)s"
# logging.basicConfig(level=logging.DEBUG, format=fmt)
# logging.basicConfig(level=logging.ERROR, format=fmt)
logging.basicConfig(level=logging.INFO, format=fmt)
###########################


import requests
import json
import urllib.parse
# import pprint
from boto3.session import Session
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

#定数定義
PUT_KEY_PRF='Torishima/Gateway/Log/Proaxia/'    # S3出力先の先頭部
# PUT_KEY_PRF='MUR/Fact/Gateway/Test/'

class Serializer(object):
    @staticmethod
    def serialize(object):
        return json.dumps(object, default=lambda o: o.__dict__.values()[0])

# def waiter_Sample():
#     #https://qiita.com/kimihiro_n/items/f3ce86472152b2676004
#     s3 = boto3.client('s3')
#     # object ができるまで待機する waiter を作成
#     waiter = s3.get_waiter('object_exists')
#     # NewObject.txt が作成されるまで sleep する
#     waiter.wait(Bucket='test_bucket', Key='NewObject.txt')    

# Cognito結果用
class CognitoResult():
    def __init__(self,res, identityId='',token='',region=''):
        self.Result=res
        self.identityId=identityId
        self.token=token
        self.region=region
    def default_method(self,item):
        if isinstance(item, object) and hasattr(item, '__dict__'):
            return item.__dict__
        else:
            raise TypeError
    def to_json(self):
        def default_method(item):
            if isinstance(item, object) and hasattr(item, '__dict__'):
                return item.__dict__
            else:
                raise TypeError
        # 作成したメソッドを指定してdumps
        return json.dumps(self, default=default_method, indent=1)

#Cognito処理
def CognitoProc(maddr,passwd,server ):
    d = { 'mailAddress': maddr, 'password' : passwd}
    d_qs = urllib.parse.urlencode(d)
    server = "https://mt-dev.tr-com.cloud/api/v1/cognitoauth"
    # server = "https://dev.tr-com.cloud/api/v1/cognitoauth"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    r = requests.post(server , data=d_qs, headers=headers)
    # logger.debug(r)
    logger.debug('Status_code:' + str(r.status_code))
    if r.status_code != 200:
        ret=CognitoResult(False)
        return ret
    rstr = json.loads(r.text)
    # pprint.info.pprint.info( rstr)
    # logger.debug(rstr)
    identityId=rstr['identityId']
    region=rstr['region']
    token=rstr['token']
    logger.debug("=========================")
    logger.debug(identityId)
    logger.debug(region)
    logger.debug("=========================")
    
    ret=CognitoResult(True,identityId,token,region)

    return ret

# セッション接続
def getSession(maddr,passwd,server):
    # Cognito認証
    cogRet=CognitoProc(maddr,passwd,server)
    if cogRet.Result == False:
        logger.debug('First Cognito NG!!!')
        return None
    logger.debug("--cogRet.region:" + cogRet.region)
    logger.debug("--cogRet.identityId:" + cogRet.identityId)
    logger.debug("--cogRet.token:" + cogRet.token)

    # Cognito認証結果からアクセスキー取得
    logger.debug("-----get_credentials_for_identity-------")
    cognito2 = boto3.client('cognito-identity',cogRet.region)
    resp = cognito2.get_credentials_for_identity(
        IdentityId=cogRet.identityId,
        Logins={
            'cognito-identity.amazonaws.com': cogRet.token
        }
    )
    # アクセスキー、シークレットキー、トークン取得
    accessKey = resp['Credentials']['AccessKeyId']
    secretKey = resp['Credentials']['SecretKey']
    logger.debug("accessKey:" + accessKey)
    logger.debug("secretKey:" + secretKey)
    token2 = resp['Credentials']['SessionToken']
    logger.debug("-----Session open s3-------")
    # アクセスキー、シークレットキー、トークンでセッション接続
    session = Session(
                aws_access_key_id=accessKey,
                aws_secret_access_key=secretKey,
                aws_session_token=token2, #トークンを忘れずに！
                region_name=cogRet.region)
    return session

# 指定されたオブジェクト（あるいはテキスト）をJSON変換してS3に送信する
def PutJsonFromObj(server,maddr,passwd,bucket,key,dataobj,isText=False):
    # セッション接続
    # ※返されたセッションでS3接続する
    logger.debug('PutJsonFromObj %s:%s:%s:%s:%s' % (maddr,passwd,bucket,key,str(isText)))

    session=getSession(maddr,passwd,server)
    if session == None:
        logger.warning("getSession Error:maddr=%s,passwd=%s,bucket=%s,key=%s" % (maddr,passwd,bucket,key))
        return False
    logger.debug("  Bucket:%s , Prefix:%s" % (bucket,key))

    logger.debug("-----s3client.Object-------")
    s3 = session.resource('s3') # セッションでS3接続
    bucket = s3.Bucket(bucket)

    obj = bucket.Object(key)
    rsp=None
    if isText == False:
        # オブジェクトの場合はJSON化
        putstr=''
        if isinstance(dataobj, list) :
            # リストなら各要素をJSON化
            for item in dataobj:
                js=item.to_json()
                putstr += '\r\n' + js
        else:
            # JSON化
            putstr=dataobj.to_json()
        # S3送信
        rsp = obj.put(Body = putstr)
    else:
        # テキストの場合（そのまま送信）送信
        rsp = obj.put(Body = dataobj)

    # 結果判定    
    if rsp['ResponseMetadata']['HTTPStatusCode']==200:
        return True
    else:
        logger.waring('Put resp HTTPStatusCode:%s' % rsp['ResponseMetadata']['HTTPStatusCode'])
        return False

# センサーデータ出力用
def SensorDataS3Put(server,maddr,passwd,bucket,dataobj,name='Adver',isText=False):
    logger.debug('SensorDataS3Put Go:maddr=%s,passwd=%s,bucket=%s,dataobj=%s,name=%s,isText=%s' % (maddr,passwd,bucket,dataobj,name,isText) )

    datestr=datetime.now().strftime('%Y%m%d%H%M%S')
    key=PUT_KEY_PRF + name + '/' + name + '_'+datestr+'.json'
    logger.debug('S3Put Key=' + key)
    # S3put
    rt=PutJsonFromObj(server,maddr,passwd,bucket,key,dataobj,isText)
    return rt

def main():
    # Test用
    #Send Data    
    dataobj2=[]
    dataobj2.append(CognitoResult(True,'id10','token1','reg1'))
    dataobj2.append(CognitoResult(True,'id20','token2','reg2'))
    dataobj2.append(CognitoResult(True,'id30','token3','reg3'))
    dataobj2.append(CognitoResult(True,'id40','token4','reg4'))

    strjson=''
    for item in dataobj2:
        strjson+=item.to_json() + '\r\n'
    logger.debug(strjson)

    PutJsonFromObj('','ryusryureg@gmail.com','1Proaxia!','torishima-dev','Torishima/Gateway/Log/GWA000001/puttest.json',dataobj2,False)


if __name__ == '__main__':
    main()
