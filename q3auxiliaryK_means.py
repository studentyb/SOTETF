from numpy import *
import pickle
#载入文件

def loadfile(): # 数据是一个大列表 其中每个元素是一个矩阵
    fileName = "D:\\MySOTorrentData\\docvecs\\vec.pkl"
    with open(fileName,"rb") as f:
        data = pickle.load(f)
    return data # data是一个大列表包含50549条数据 其中每条数据是一个1*50的矩阵

#计算向量距离
def distEclud(vecA,vecB,ax = 1):
    # return np.linalg.norm(vecA - vecB, axis = ax)
    return sqrt(sum(power(vecA - vecB, 2)))

#构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    n = shape(dataSet)[1] #计算列数
    print("data的列数n为:{}".format(n))
    #创建一个k行n列的矩阵用于保存k个质心 一行一个质心
    centroid = mat(zeros((k,n))) #mat是numpy中的函数 可以将数组转换成矩阵
    for j in range(n): # n=50列
        min_j = min(dataSet[:,j]) #取每列最小值
        range_j = float(max(dataSet[:,j]) - min_j)
        centroid[:,j] = min_j + range_j*random.rand(k,1) #构建k行n列，每行代表质心坐标
    return centroid

def k_means(dataSet, k, distMeas = distEclud, creatCent = randCent):
    m = shape(dataSet)[0] #行数
    print("data的行数为:{}".format(m))
    clusterAssment = mat(zeros((m,2))) # 建立簇分配结果矩阵，第一列存索引，第二列存误差
    # zeros((m,2))生成m行2列的数组 再用mat转换成矩阵
    #numpy.zeros(shape，dtype = float，order = 'C')
    #shape接受一个int参数或者一个int元组
    centroids = creatCent(dataSet, k) #聚类点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #针对每一行
            minDist = inf # 无穷大
            minIndex = -1 # 初始化
            for j in range(k): #对于每一个聚类中心
                #计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist: # 存储最小值，存储最小值所在的位置
                    minDist = distJI
                    minIndex = j #记录此行属于哪个聚类
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print("分类后四个中心的位置:")
        print(centroids)

        for cent in range(k): #4个质心
            ptsInClust = []
            for i in range(m): # 每一行
                if clusterAssment[i,0] == cent:#如果该行的类别属于当前质心所属的类别
                    ptsInClust.append((dataSet[i].tolist())[0]) #tolist将数组或者矩阵转换成列表
                    #将每一行数据转换成列表 取第一个数 放置于ptsInClust中
            ptsInClust_m = mat(ptsInClust)
            print(mean(ptsInClust,axis=0))
            centroids[cent,:] = mean(ptsInClust_m,axis=0) #沿矩阵列方向进行均值计算 重新计算质心
                                            #axis = 0 压缩行 对各列求均值 返回1行x列的矩阵
        print("此时各个类别包含的元素数量:")
        c0= c1 = c2 = c3 =0
        for i in range(len(clusterAssment)):
                if clusterAssment[i,0] == 0:
                    c0 = c0 + 1
                if clusterAssment[i,0] == 1:
                    c1 = c1 + 1
                if clusterAssment[i,0] == 2:
                    c2 = c2 + 1
                if clusterAssment[i,0] == 3:
                    c3 = c3 + 1
        print("c0:{} c1:{} c2:{} c3:{}".format(c0,c1,c2,c3))
    return centroids,clusterAssment

if __name__ == "__main__":

    dataMat = mat(loadfile())
    myCentroids,clustAssing = k_means(dataMat, 3)
    with open("D:/MySOTorrentData/docvecs/clustAssing.pkl","wb") as fd:
        pickle.dump(clustAssing,fd)
        
    # 便于用记事本打开 复制到excel中查看分析
    fc = open("D:/MySOTorrentData/classfy.txt", "a", encoding="utf-8")
    for i in range(len(clustAssing)):
        print(i)
        mystr = str(int(clustAssing[i, 0])) + "\n"
        fc.write(mystr)
    fc.close()

    with open("D:/MySOTorrentData/docvecs/Centroids.pkl","wb") as fc:
        pickle.dump(myCentroids,fc)
    print("finish processing")