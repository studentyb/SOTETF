import datetime # 使用datetime库 可以计算两个时间的差值
import pickle

PostAllEdit = [] #所有post的所有edit
EditId = [] # 每一个postId的所有edit的Id
middVar = []
postId = [] #记录所有post的Id
editAllId = []
timeDiffList = []
posts = ""
edits = ""
answerCount = []
def processData():

    global posts
    global edits
    # 打开文件
    fPost = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    fEdit = "D:\\MySOTorrentData\\soted_by_PostId\\postHistory_2014_sorted.pkl"
    with open(fPost,"rb") as fp:
        posts = pickle.load(fp)
    with open(fEdit,"rb") as fe:
        edits = pickle.load(fe)

    #获得所有post的Id 保存到列表postId中
    for i in range(len(posts)):
        postId.append(posts[i]["Id"])

    #获得所有edit的Id 保存到列表editAllId中
    for i in range(len(edits)):
        editAllId.append(edits[i]["Id"])

    #对于每一个post的Id 获得其所有的edit的Id保存为一个列表
    #所有post的edit信息保存在posts这个大列表中
    #posts是一个二维数据
    for id in postId:
        for edit in edits:
            if id == edit["PostId"]:
                # middVar.append(eval(edit["Id"])) 不修改id为整数型 保持字符型
                middVar.append(edit["Id"])
        # PostAllEdit.append(middVar) # append函数 当middVar中的内容改变时 之前已经添加的都会随之一起改变
        PostAllEdit.append((middVar[:]))
        middVar.clear()

    for i in range(len(postId)):
        initDate = ""
        editDate = ""
        if len(PostAllEdit[i]) != 0: # 有修改记录
            for edit in edits:
                if edit["Id"] == PostAllEdit[i][0]:
                     editDate = edit["CreationDate"]
                     break
            for post in posts:
                if post["Id"] == postId[i]:
                   initDate = post["CreationDate"]
            initDate = initDate[:19]
            editDate = editDate[:19] # 使用字符串的切片方式去除秒之后的多余字符
            edit_Date = datetime.datetime.strptime(editDate,"%Y-%m-%dT%H:%M:%S")
            init_Date = datetime.datetime.strptime(initDate,"%Y-%m-%dT%H:%M:%S")
            # timediff = (edit_Date - init_Date).hours 没有hours属性 只有days属性
            timediff = (edit_Date - init_Date).total_seconds() / 3600 # 换算成以小时为单位
            timeDiffList.append(round(timediff,5)) # 浮点数的尾数有不确定性 用round函数去除多余尾数 保持精度
    diff_1 = 0
    diff_2 = 0
    diff_3 = 0
    diff_4 = 0
    diff_5 = 0
    for i in range(len(timeDiffList)):
        if timeDiffList[i] > 720: # 大于一个月
            diff_1 = diff_1 + 1
        elif timeDiffList[i] > 168: # 大于一周小于一个月
            diff_2 = diff_2 + 1
        elif timeDiffList[i] > 24: # 大于一天小于一周
            diff_3 = diff_3 +1
        elif timeDiffList[i] > 1: # 大于一个小时小于一天
            diff_4 = diff_4 + 1
        else : # 小于一小时
            diff_5 = diff_5 + 1

    print("总共有{}个时间差".format(len(timeDiffList)))
    print("一个月以上一年以内edit共{}个".format(diff_1))
    print("一周以上一个月以内edit共{}个".format(diff_2))
    print("一天以上一周以内edit共{}个".format(diff_3))
    print("一小时以上一天以内edit共{}个".format(diff_4))
    print("一小时以内edit共{}个".format(diff_5))

def AnswerCount():
    unedit = 0
    count_un = 0
    edit = 0
    count = 0
    for post in posts:  # AnswerCount 和 postAllEdit一一对应
        answerCount.append(post.get("AnswerCount",0)) # 若键存在 返回相应值否则返回0
    for i in range(len(PostAllEdit)):
        if len(PostAllEdit[i]) == 0:
            unedit = unedit + 1
            count_un = count_un + eval(answerCount[i])
        else:
            edit = edit + 1
            count = count + eval(answerCount[i])
    print("编辑过的问题总共:{}个".format(edit))
    print("编辑过的问题获得的总的回答数:{}".format((count)))
    print("#################")
    print("没有编辑过的问题总共:{}个".format(unedit))
    print("没有编辑过的问题获得的总的回答数:{}".format(count_un))

if __name__ == "__main__":
    print("Question one:")
    processData()
    print("Question two:")
    AnswerCount()