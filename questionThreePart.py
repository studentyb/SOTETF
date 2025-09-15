import pickle
import nltk # 把anconda配置环境变量之后解决了ddl问题 下载一个nltk_data压缩包解压在C盘根目录下


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
    answerfile = "D:\\MySOTorrentData\\soted_by_PostId\\answer_2014_sorted.pkl"
    answers = loadfile(answerfile)
    posts = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    data = loadfile(posts)
    for post in data:
        # postId和AcceptedAnswerId一一对应 相等索引
        postId.append(post["Id"])
        AAId.append(post["AcceptedAnswerId"])
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
    return answersTime

def part_overall(edits):
    title = 0  # PostHistoryTypeId = 4
    tag = 0  # PostHistoryTypeId = 6
    body = 0  # PostHistoryTypeId = 5

    for edit in edits:
        if edit["PostHistoryTypeId"] == '4':
            title = title + 1
        if edit["PostHistoryTypeId"] == '5':
            body = body + 1
        if edit["PostHistoryTypeId"] == '6':
            tag = tag + 1
    print("共有edit{}个".format(title+body+tag)) # 11772以下三项相加和总数不符 因为有rollback
    print("编辑title部分:{}个".format(title))
    print("编辑body部分:{}个".format(body))
    print("编辑tag部分:{}个".format(tag))

def who_overall(edits):
    owner = []
    postId = []
    postOwner = 0
    other = 0
    posts = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    data = loadfile(posts)
    for post in data:
        owner.append(post.get("OwnerUserId","0"))
        postId.append(post["Id"])
    for edit in edits:
        editUser = edit.get("UserId",'0')
        for i in range(len(postId)):
            if(edit["PostId"]) != postId[i]:
                continue
            if editUser == owner[i]:
                postOwner = postOwner + 1
            else:
                other = other + 1
    print("总的编辑次数{}".format(len(edits)))
    print("Owner做的编辑{}次".format(postOwner))
    print("Other做的编辑{}次".format(other))

def part_distribution(edits):

    fat = fab = fatg = 0
    aat = aab = aatg = 0
    aa_t = aa_b = aa_tg = 0
    lat = lab =latg =0

    title_all = 0
    body_all = 0
    tag_all = 0
    index = 0
    postId = []

    timePoint = getAnswer()
    posts = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    data = loadfile(posts)
    for post in data:
        # 获得所有postID
        postId.append(post["Id"])
    for edit in edits:
        Date = edit["CreationDate"]
        editDate = Date[:19] # 当前edit发生的时间
        editPostId = edit["PostId"] # 当前edit编辑的post
        editType = edit["PostHistoryTypeId"]
        for i in range(len(postId)):
            if editPostId == postId[i]:
                index = i # 取索引
                break

        FA = timePoint[index]["FA"]
        AA = timePoint[index]["AA"]
        LA = timePoint[index]["LA"]

        if FA != "null":
            if editDate < FA:
                if editType == '4': #编辑title
                    fat = fat + 1
                if editType == '5': #编辑body
                    fab = fab + 1
                if editType == '6': #编辑tag
                    fatg = fatg + 1

        if AA != "null":
            if editDate < AA :
                if editType == '4': #编辑title
                    title_all = title_all + 1
                    aat = aat + 1
                if editType == '5': #编辑body
                    body_all = body_all + 1
                    aab = aab + 1
                if editType == '6': #编辑tag
                    tag_all = tag_all + 1
                    aatg = aatg + 1

        if AA != "null":
            if editDate > AA :
                if editType == '4': #编辑title
                    title_all = title_all + 1
                    aa_t = aa_t + 1
                if editType == '5': #编辑body
                    body_all = body_all + 1
                    aa_b = aa_b + 1
                if editType == '6': #编辑tag
                    tag_all = tag_all + 1
                    aa_tg = aa_tg + 1

        if LA != "null":
            if editDate > LA :
                if editType == '4': #编辑title
                    lat = lat + 1
                if editType == '5': #编辑body
                    lab = lab + 1
                if editType == '6': #编辑tag
                    latg = latg + 1

    print("编辑title的总次数{}".format(title_all))
    print("编辑body的总次数{}".format(body_all))
    print("编辑tag的总次数{}".format(tag_all))
    print("<FA:")
    print("title:{} body:{} tag{}".format(fat,fab,fatg))
    print("<AA:")
    print("title:{} body{} tag{}".format(aat,aab,aatg))
    print(">AA:")
    print("title:{} body{} tag{}".format(aa_t,aa_b,aa_tg))
    print(">LA:")
    print("title:{} body{} tag{}".format(lat,lab,latg))

def who_distribution(edits):

    faowner = aaowner = aa_owner = laowner = 0
    fauser = aauser = aa_user = lauser = 0
    owner_all = 0
    user_all = 0
    index = 0
    owner = ""
    timePoint = getAnswer()
    posts = "D:\\MySOTorrentData\\soted_by_PostId\\post_2014_sorted.pkl"
    data = loadfile(posts)

    for edit in edits:
        editPer = edit.get("UserId")
        editPost = edit.get("PostId")
        Date = edit["CreationDate"]
        editDate = Date[:19]  # 当前edit发生的时间
        for i in range(len(data)):
            if editPost == data[i]["Id"]:
                index = i
                owner = data[i].get("OwnerUserId")
                break
        FA = timePoint[index]["FA"]
        AA = timePoint[index]["AA"]
        LA = timePoint[index]["LA"]

        if FA != "null":
            if editDate < FA:
                if editPer == owner:
                    faowner = faowner + 1
                else:
                    fauser = fauser + 1

        if AA != "null":
            if editDate < AA :
                if editPer == owner:
                    owner_all = owner_all + 1
                    aaowner = aaowner + 1
                else:
                    user_all = user_all + 1
                    aauser = aauser + 1
        if AA != "null":
            if editDate > AA :
                if editPer == owner:
                    owner_all = owner_all + 1
                    aa_owner = aa_owner + 1
                else:
                    user_all = user_all + 1
                    aa_user = aa_user + 1
        if LA != "null":
            if editDate > LA :
                if editPer == owner:
                    laowner = laowner + 1
                else:
                    lauser = lauser + 1

    print("Owner编辑的总次数:{}".format(owner_all))
    print("User编辑的总次数:{}".format(user_all))
    print("<FA:")
    print("Owner:{}".format(faowner))
    print("User:{}".format(fauser))
    print("<AA")
    print("Owner:{}".format(aaowner))
    print("User:{}".format(aauser))
    print(">AA")
    print("Owner:{}".format(aa_owner))
    print("User:{}".format(aa_user))
    print(">LA")
    print("Owner:{}".format(laowner))
    print("User:{}".format(lauser))

if __name__ == "__main__":
    posthistory = "D:\\MySOTorrentData\\soted_by_PostId\\postHistory_2014_sorted.pkl"
    data = loadfile(posthistory)
    print("who edits:")
    who_overall(data)
    print("what part is edited:")
    part_overall(data)
    print("####")
    print("part distribution:")
    part_distribution(data)
    print("who distribution")
    who_distribution(data)