# postHistoryTypeId in [4,5,6,7,8,9] 编辑
# 将post载入 获得post.Id
# post.Id == PostId
# 读取postHistory

import re
import time
import pickle
count = 0

marktxt = ['﻿<?xml version="1.0" encoding="utf-8"?>\n', '<posthistory>\n', '</posthistory>']
historyType = ['4','5', '6', '7', '8', '9']
postsEdit = []
edit = {}
postId = []
def readFile(fileName):
    global count
    global lastId
    global postsEdit
    regex = re.compile(r'\b(.[^=]*)="(.[^"]*)"')
    with open(fileName,"rt",encoding="UTF-8") as file:
        for line in file:
            if line in marktxt:
                continue
            edit = dict(regex.findall(line))
            edit["Id"] = edit.pop("row Id")
            if edit["CreationDate"] < "2014-01-01":
                count = count + 1
                continue
            if edit["CreationDate"] > "2014-12-31":
                break
            if edit["PostId"] in postId:
                if edit["PostHistoryTypeId"] in historyType:
                    postsEdit.append(edit)
                    print("Add into postEdit")

def getPostId(fileName):
    with open(fileName,'rb') as f:
        data = pickle.load(f)
        for item in data:
            postId.append(item["Id"])

if __name__ == "__main__":

    filePost = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    filePostHistory = "D:\\SOTorrentDataBase_2\\PostHistory.xml\\PostHistory.xml"
    dumpName = "D:\\MySOTorrentData\\soted_by_PostId\\postHistory_2014_sorted.pkl.pkl"

    getPostId(filePost) # 获得postId的列表
    print("postId:" + str(postId))
    start_match = time.perf_counter()
    readFile(filePostHistory) # 根据postId 读取post的所有编辑
    data = postsEdit
    end_match = time.perf_counter()
    print("共跳过{}".format(count))
    print("匹配edit总耗时:{:.2f}s".format(end_match-start_match))
    data.sort(key=lambda x:eval(x["PostId"]))
    with open(dumpName,"wb") as fh:
        pickle.dump(postsEdit,fh)
    print("处理完毕")