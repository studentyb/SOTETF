# 重复调用 分别获得2015-2018的数据
# postTypeId = 1 即question
# AcceptedAnswerId != null 即 question存在acceptedAnswer
# CreationDate > 2015-01-01 && CreationDate < 2015-12-31
# Score != 0

import  re
import time
import pickle
result = [] # 整个文件数据组织成一个列表
item = {} # 列表中每一个元素是一个字典
count = 0
markTxt = ('﻿<?xml version="1.0" encoding="utf-8"?>\n', '<posts>\n', '</posts>')
def getPost(fileName):
    global count
    regex = re.compile(r'\b(.[^=]*)="(.[^"]*)"')
    with open(fileName, "rt", encoding="UTF-8") as file:
        for line in file:
            if line in markTxt: # 去除其他信息
                continue
            item = dict(regex.findall(line))
            item["Id"] = item.pop("row Id")
            if(eval(item["Score"]) != 0):
                if "AcceptedAnswerId" in item:
                    if(eval(item["PostTypeId"]) == 1):
                        if ( (item["CreationDate"] > "2015-01-01") and  \
                             (item["CreationDate"] < "2015-12-31") ):
                            if( count <= 10000):
                                count = count + 1
                                result.append(item)
                                print("第{}条".format(count))
	return result

if __name__ == "__main__":
    fileName = "F:\\SOTorrentDataBase\\Posts.xml\\Posts.xml"
    dumpName = "D:\\MySOTorrentData\\soted_by_PostId\\post_2015_sorted.pkl"
    start_filter = time.perf_counter()
    data = getPost(fileName)
    end_filter = time.perf_counter()
    print("过滤总耗时:{:.2f}秒".format(end_filter - start_filter))
    data.sort(key=lambda x:eval(x["Id"])) #按ID排序
    start_dump = time.perf_counter()
    with open(dumpName, "wb") as fh:
        pickle.dump(result, fh)
    end_dump = time.perf_counter()
    print("存储总耗时:{:.2f}秒".format(end_dump-start_dump)) #格式输出时必须有冒号