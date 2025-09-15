import pickle

def jointfile(): #分别将每年的post postHistory editComment连接成一个文件
    comment = []
    for i in range(2014,2019):
        print(i)
        filename = "D:/MySOTorrentData/soted_by_PostId/postHistory_{}_sorted.pkl".format(i)
        with open(filename,"rb") as file:
            data = pickle.load(file)
        for item in data:
            comment.append(item.get("Comment","null"))
    print("comment:{}".format(len(comment)))
    with open("D:/MySOTorrentData/soted_by_PostId/comment.pkl","wb")as f:
        pickle.dump(comment,f)

def loadfile(fileName):
    with open(fileName,"rb") as file:
        data = pickle.load(file)
    return data

def getAnswer():
    postId = []
    AAId = []
    timeDic = {}
    midvar = []
    fpart = []
    spart = []
    answersTime = []
    # 获得每个Post的answer
    answerfile = "D:\\MySOTorrentData\\soted_by_PostId\\answer.pkl"
    answers = loadfile(answerfile)
    print("num of answers:{}".format(len(answers)))
    posts = "D:\\MySOTorrentData\\soted_by_PostId\\post.pkl"
    data = loadfile(posts)
    print("num of posts:{}".format(len(data)))
    print("now,get all the post ID")
    for post in data:
        # postId和AcceptedAnswerId一一对应 相等索引
        postId.append(post["Id"])
        AAId.append(post["AcceptedAnswerId"])
    print("finish get post ID")
    print("###")
    print("now,get all answer time dic")
    for i in range(len(postId)):
        midvar.clear()
        for answer in answers:
            if answer.get("ParentId","0") == postId[i] :
                date = answer["CreationDate"][:19]
                midvar.append(date)
        #对于每一个post的answer 若midvar长度等于1 则是AcceptedAnswer
        if len(midvar) == 1:
            timeDic["PostId"] = postId[i]
            timeDic["AA"] = midvar[0]
            timeDic["FA"] = "null"
            timeDic["LA"] = "null"
        else:
            # post有两个及以上answer
            fpart.clear()
            spart.clear()
            timeDic.clear()
            timeDic["PostId"] = postId[i]
            for answer in answers:
                if AAId[i] == answer["Id"]: # post的AcceptedAnswerID
                    timeDic["AA"] = answer["CreationDate"][:19]
                    break
            for i in range(len(midvar)):
                if(midvar[i] < timeDic["AA"]):
                    fpart.append(midvar[i])
                elif (midvar[i] > timeDic["AA"]):
                    spart.append(midvar[i])
                else:
                    continue
            if len(fpart) == 0:
                timeDic["FA"] = "null"
            else:
                timeDic["FA"] = fpart[0]

            if len(spart) == 0:
                timeDic["LA"] = "null"
            else:
                timeDic["LA"] = spart[-1]
        answersTime.append(timeDic.copy())
    # print(len(answersTime))
    print("finish get all answer time dic")
    print("the length of answersTime:{}".format(len(answersTime)))
    return answersTime,postId

def type_distribution(edits):
    count = 0
    index = 0
    timePoint,postId = getAnswer()
    classfy = []
    fac0 = fac1 = fac2 = 0
    aac0 = aac1 = aac2 = 0
    aa_c0 = aa_c1 = aa_c2 = 0
    lac0 = lac1 = lac2 = 0
    with open("D:/MySOTorrentData/classfy.txt", "r", encoding="utf-8") as fc:
        data = fc.readlines()
        for item in data:
            #获得所有comment的分类
            classfy.append(item.strip())
    print("the length of classfy:{}".format(len(classfy)))
    print("now,get result")
    for edit in edits:
        if edit.get("PostHistoryTypeId") not in ['4','5']:
            continue
        editPost = edit.get("PostId")
        Date = edit["CreationDate"]
        editDate = Date[:19]  # 当前edit发生的时间
        for i in range(len(postId)):
            if editPost == postId[i]:
                index = i
                break
        FA = timePoint[index]["FA"]
        AA = timePoint[index]["AA"]
        LA = timePoint[index]["LA"]
        print("count:{}".format(count))
        if FA != "null":
            if editDate < FA :
                if classfy[count] == '0':
                    fac0 += 1
                if classfy[count] == '1':
                    fac1 += 1
                if classfy[count] == '2':
                    fac2 += 1

        if AA != "null":
            if editDate < AA:
                if classfy[count] == '0':
                    aac0 += 1
                if classfy[count] == '1':
                    aac1 += 1
                if classfy[count] == '2':
                    aac2 += 1

        if AA != "null":
            if editDate > AA:
                if classfy[count] == '0':
                    aa_c0 += 1
                if classfy[count] == '1':
                    aa_c1 += 1
                if classfy[count] == '2':
                    aa_c2 += 1

        if LA != "null":
            if editDate > LA:
                if classfy[count] == '0':
                    lac0 += 1
                if classfy[count] == '1':
                    lac1 += 1
                if classfy[count] == '2':
                    lac2 += 1
        count += 1
    print("count:{}".format(count))
    print("<FA:")
    print("type0:{}".format(fac0))
    print("type1:{}".format(fac1))
    print("type2:{}".format(fac2))
    print("<AA:")
    print("type0:{}".format(aac0))
    print("type1:{}".format(aac1))
    print("type2:{}".format(aac2))
    print(">AA:")
    print("type0:{}".format(aa_c0))
    print("type1:{}".format(aa_c1))
    print("type2:{}".format(aa_c2))
    print(">LA:")
    print("type0:{}".format(lac0))
    print("type1:{}".format(lac1))
    print("type2:{}".format(lac2))

if __name__ == "__main__":
    posthistory = "D:/MySOTorrentData/soted_by_PostId/postHistory.pkl"
    data = loadfile(posthistory)
    type_distribution(data)
    # jointfile()