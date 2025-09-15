import pickle

def filterEdit():
    data = []
    fedit = "D:/MySOTorrentData/soted_by_PostId/postHistory.pkl"
    with open(fedit,"rb") as fe:
        edits = pickle.load(fe)
    for edit in edits:
        if edit["PostHistoryTypeId"] in ['7','8','9']:
            continue
        data.append(edit)
    return data #61030

def getEidtPostId(): #过滤后的postId
    postId = []
    edits = filterEdit()
    for edit in edits:
        postId.append(edit["PostId"])
    return postId

def getOwnerId():
    fpost = "D:/MySOTorrentData/soted_by_PostId/post.pkl"
    ownerId = []
    with open(fpost,"rb") as fp:
        posts = pickle.load(fp)
    postId = getEidtPostId()
    for i in range(len(postId)):
        for post in posts:
            if postId[i] == post["Id"]:
                ownerId.append(post.get("OwnerUserId"))
                break
    return ownerId #61030

def filterAnswer():
    count = 0
    cur = []
    postId = getEidtPostId()
    fanswer = "D:/MySOTorrentData/soted_by_PostId/answer.pkl"
    with open(fanswer, "rb") as fa:
        answers = pickle.load(fa)
    print(len(answers))
    for answer in answers:
        if answer["ParentId"] not in postId:
            continue
        else:
            cur.append(answer)
            count += 1
            if count %1000 == 0:
                print(count)
    with open("D:/MySOTorrentData/answerAfterFilter.pkl","wb") as f:
        pickle.dump(cur,f)
    print(len(cur))

def getAnswerTime():
    cur = []
    answerTime = []
    postId = getEidtPostId()
    fanswer = "D:/MySOTorrentData/answerAfterFilter.pkl"
    with open(fanswer,"rb") as fa:
        answers = pickle.load(fa)
    for i in range(len(postId)):
        cur.clear()
        for answer in answers:
            if answer["ParentId"] != postId[i]:
                continue
            else:cur.append(answer["CreationDate"])
        answerTime.append(cur.copy())
    return answerTime #61030

def getFeature():
    edits = filterEdit()
    ownerId = getOwnerId()
    answerTime = getAnswerTime()
    feature = []
    features = []
    typeId = []
    fclass = "D:/MySOTorrentData/classfy.txt"
    editIndex = 0
    with open(fclass,"rt") as fc:
        classfier = fc.readlines()
    for index,edit in enumerate(edits):
        count = 0
        feature.clear()
        if edit["PostHistoryTypeId"] == '4':
            typeId.append(index)
            feature.append(1)
            feature.append(0)
            feature.append(0)
            if edit.get("UserId") == ownerId[index]:
                feature.append(1)
                feature.append(0)
            else:
                feature.append(0)
                feature.append(1)
            for i in range(len(answerTime[index])):
                if edit["CreationDate"] <= answerTime[index][i]:
                    count += 1
            feature.append(count)
            if classfier[editIndex] == '0\n':
                print("进入")
                feature.append(1)
                feature.append(0)
                feature.append(0)
            if classfier[editIndex] == '1\n':
                feature.append(0)
                feature.append(1)
                feature.append(0)
            if classfier[editIndex] == '2\n':
                feature.append(0)
                feature.append(0)
                feature.append(1)
            editIndex += 1
        if edit["PostHistoryTypeId"] == '5':
            typeId.append(index)
            feature.append(0)
            feature.append(1)
            feature.append(0)
            if edit.get("UserId") == ownerId[index]:
                feature.append(1)
                feature.append(0)
            else:
                feature.append(0)
                feature.append(1)
            for i in range(len(answerTime[index])):
                if edit["CreationDate"] <= answerTime[index][i]:
                    count += 1
            feature.append(count)
            if classfier[editIndex] == '0\n':
                feature.append(1)
                feature.append(0)
                feature.append(0)
            if classfier[editIndex] == '1\n':
                feature.append(0)
                feature.append(1)
                feature.append(0)
            if classfier[editIndex] == '2\n':
                feature.append(0)
                feature.append(0)
                feature.append(1)
            editIndex += 1
        if edit["PostHistoryTypeId"] == '6':
            feature.append(0)
            feature.append(0)
            feature.append(1)
            if edit.get("UserId") == ownerId[index]:
                feature.append(1)
                feature.append(0)
            else:
                feature.append(0)
                feature.append(1)
            for i in range(len(answerTime[index])):
                if edit["CreationDate"] <= answerTime[index][i]:
                    count += 1
            feature.append(count)
            feature.append(0)
            feature.append(0)
            feature.append(0)
        features.append(feature.copy())
    print("editIndex{}".format(editIndex))
    print("typeId共有{}条数据".format(len(typeId)))
    print("总共有{}条数据".format(len(features)))
    with open("D:/MySOTorrentData/features.pkl","wb") as ff:
        pickle.dump(features,ff)
    return features

if __name__ == "__main__":
    getFeature()