import pickle
classfy = []
with open("D:/MySOTorrentData/classfy.txt", "r", encoding="utf-8") as fc:
    data = fc.readlines()
    for item in data:
        # 获得所有comment的分类
        classfy.append(item.strip())
print("the length of classfy:{}".format(len(classfy)))
with open("D:/MySOTorrentData/soted_by_PostId/comment.pkl","rb")as f:
    comment = pickle.load(f)
print("comment:{}".format(len(comment)))
with open("D:/MySOTorrentData/soted_by_PostId/postHistory.pkl","rb") as fp:
    posts = pickle.load(fp)
print("posts:{}".format(len(posts)))
count = 0
for item in posts:
    if(item.get("PostHistoryTypeId") not in ['4','5']):
        continue
    else:
        count += 1
print("num of type4 and type5:{}".format(count))

#运行结果：
#the length of classfy:50549
#comment:50549
#posts:61418
#num of type4 and type5:50549