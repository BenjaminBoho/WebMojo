import sys

import trcom
import trend
import battery
import traceback

#メイン処理
def main():

    params = initcommon()
    brainresult = params.params["brainresult"]

    if (params.guid == "braintest01"):
        if params.params["exectrend"] == True:
            trend.trend_plot(params)
        if params.params["execbattery"] == True:
            battery.battery_trend(params, params.params["brainresult"])
    else:
        try:
            if params.params["exectrend"] == True:
                trend.trend_plot(params)
            if params.params["execbattery"] == True:
                battery.battery_trend(params, params.params["brainresult"])
        except Exception as e:
            brainresult["error"] = "PythonRuntimeErrors"
            brainresult["errordetail"] = str(e) + "\n" + traceback.format_exc()
            print(e)

    params.save_result(params.params["brainresult"])

def initcommon():
    args = sys.argv
    params = trcom.params(args)
    prm = params.params
    verylonghours = prm["verylonghours"]
    brainresult = {"files":[], "output":{}}
    prm["brainresult"] = brainresult

    brainresult["output"]["lifetime1"] = {}
    brainresult["output"]["lifetime2"] = {}
    brainresult["output"]["lifetime1min"] = verylonghours
    brainresult["output"]["lifetime2min"] = verylonghours
    brainresult["output"]["lifetime1max"] = verylonghours
    brainresult["output"]["lifetime2max"] = verylonghours
    brainresult["output"]["outliers"] = [];
    brainresult["output"]["lasttrenddate"] = ""
    brainresult["output"]["nonewdata"] = False
    brainresult["output"]["batterydeterioration"] = False
    brainresult["output"]["batteryslope"] = 0
    brainresult["output"]["tholdmessages"] = []
    trcom.font_setup()
    return params

if __name__=='__main__':
    main()
