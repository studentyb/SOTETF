import time
import pickle
import re

AllAnswers = [] #用于存放所有的answer
answer = {} # 用于存放一个answer
postId = []
marktxt = ['﻿<?xml version="1.0" encoding="utf-8"?>\n', '<posthistory>\n', '</posthistory>']
def getAnswer():
    postIds = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    with open(postIds,"rb") as file:
        postIds = pickle.load(file)
    #获得所有待查postId
    for item in postIds:
        postId.append(item["Id"])  # post已按id进行了排序

    posts = "F:\\SOTorrentDataBase\\Posts.xml\\Posts.xml"
    regex = re.compile(r'\b(.[^=]*)="(.[^"]*)"')
    with open(posts,"rt",encoding="UTF-8") as f:
        for index,line in enumerate(f):
            if index % 100000 == 0 :
                print("正在处理第{}条数据".format(index))
            if line in marktxt:
                 continue
            answer = dict(regex.findall(line))
            if answer.get('PostTypeId','0') != '2':
                continue
            elif answer.get('CreationDate','2020-01-01T00:00:00') < "2014-01-01":
                continue
            else:
                if answer.get('ParentId','0') in postId:
                    answer['Id'] = answer.pop('row Id')
                    AllAnswers.append(answer)
	return AllAnswers



if __name__ == "__main__":
    start = time.perf_counter()
    data = getAnswer()
    end = time.perf_counter()
    print("收集所有回答所用时间:{}".format(end - start))
    data.sort(key=lambda x:eval(x["ParentId"])) # 按ParentId排序 即PostId
    dumpName = "D:\\MySOTorrentData\\soted_by_PostId\\answer_2014_sorted.pkl"
    with open(dumpName,"wb") as fh:
        pickle.dump(AllAnswers,fh)
    print("处理完毕")