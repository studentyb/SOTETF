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
    fPost = "D:\\MySOTorrentData\\soted_by_PostId\\post.pkl"
    fEdit = "D:\\MySOTorrentData\\soted_by_PostId\\postHistory.pkl"
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


def AnswerCount():
    unedit = 0
    count_un = 0
    edit = 0
    count = 0
    variance_edit = 0
    variance_unedit = 0
    for post in posts:  # AnswerCount 和 postAllEdit一一对应
        answerCount.append(post.get("AnswerCount",0)) # 若键存在 返回相应值否则返回0
    print("postAllEdit(每一项为每个post的编辑历史)长度为:{}".format(len(PostAllEdit)))
    for i in range(len(PostAllEdit)):
        if len(PostAllEdit[i]) == 0: #未被编辑过的问题
            unedit = unedit + 1
            count_un = count_un + eval(answerCount[i])
            variance_unedit += pow((eval(answerCount[i])-1.66),2) #计算方差
        else: #经过编辑的问题
            edit = edit + 1
            count = count + eval(answerCount[i])
            variance_edit += pow((eval(answerCount[i])-1.80),2)
    variance_unedit = pow(variance_unedit / unedit, 0.5)
    variance_edit = pow(variance_edit / edit, 0.5)
    print("编辑过的问题方差:{:.2f}".format(variance_edit))
    print("未经编辑的问题方差:{:.2f}".format(variance_unedit))
    print("编辑过的问题总共:{}个".format(edit)) #28562
    print("编辑过的问题获得的总的回答数:{}".format((count))) #51456 mean=1.80
    print("#################")
    print("没有编辑过的问题总共:{}个".format(unedit)) #21443
    print("没有编辑过的问题获得的总的回答数:{}".format(count_un)) #35580 mean=1.66

if __name__ == "__main__":

    processData()
    print("U verify:")
    AnswerCount()