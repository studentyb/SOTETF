from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pandas as pd
import pickle
def changeFormat():
    with open("D:/MySOTorrentData/features.pkl", "rb") as file:
        mylist = pickle.load(file)
    name = ['title', 'body', 'tag', 'owner', 'other', 'count', 'type1', 'type2', 'type3']
    test = pd.DataFrame(columns=name, data=mylist)
    test.to_csv("D:/MySOTorrentData/features.csv", encoding="gbk", index=False)

def RadomForest():
    df = pd.read_csv("D:/MySOTorrentData/features.csv")
    df.columns = ['title','body','tag','owner','other','count','type1','type2','type3']
    df.info()
    x, y = df.iloc[:,1:].values,df.iloc[:,0].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0)
    feat_labels = df.columns[1:]
    forest = RandomForestClassifier(n_estimators=100,random_state = 0,n_jobs = -1)
    forest.fit(x_train,y_train)
    feat_labels = df.columns[1:]
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %.2f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

if __name__ == "__main__":
    RadomForest()