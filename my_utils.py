import os
import shutil

def setList(listt, value):
    if (value not in listt) and (value != u""):
        listt.append(value)
    return listt

def setMap(keyValueListMap, key, value):
    valueList = keyValueListMap.get(key)
    if valueList == None:
        valueList = list()
        keyValueListMap[key] = valueList
    valueList = setList(valueList, value)
    return keyValueListMap

def setMapMap(keyValueListMap, key1, key2, value):
    valueList = keyValueListMap.get(key1)

    if valueList == None:
        valueList = {}
        keyValueListMap[key1] = valueList

    setMap(valueList, key2, value)
    return


def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)
