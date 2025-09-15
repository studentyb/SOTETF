import gensim
from gensim.models.doc2vec import Doc2Vec
import pickle
import nltk
import html
import re

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def process_raw_data():
    words = []
    raw_data = "D:\\MySOTorrentData\\soted_by_PostId\\comment.pkl"
    with open(raw_data,'rb') as file:
        data = pickle.load(file)

    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.snowball.EnglishStemmer()

    for item in data:
        comment = html.unescape(item).lower()
        for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_\'{|}~': #去除标点符号
            comment = comment.replace(ch," ")
        comment = re.sub(
            r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', 'URL', comment,
            flags=re.S)  # 除去http
        comment = re.sub(r'[~\\.\w]*(/[^ ]*)+', 'PATH', comment, flags=re.S)  # 除去url
        characters = nltk.word_tokenize(comment)
        characters = [stemmer.stem(word) for word in characters if word not in stop_words]
        # charcters是一个列表
        words.append(characters.copy())
    return words

def get_data():
    x_train = []
    docs = process_raw_data()
    for i, text in enumerate(docs):
        document = TaggededDocument(text,tags=[i])
        x_train.append(document)
    return x_train

def train(x_train,size = 50,epoch_num = 1):
    model_dm = Doc2Vec(x_train, min_count=1, window=5, vector_size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('D:\\MySOTorrentData\\model\\model_d2v_dm.model')  ##模型保存的位置

    return model_dm

if __name__ == "__main__" :
    docvecs = []
    x_train = get_data()
    model_dm = train(x_train)
    for i in range(len(model_dm.docvecs)):
        docvecs.append(model_dm.docvecs[i])
    with open("D:/MySOTorrentData/docvecs/vec.pkl","wb") as f:
        pickle.dump(docvecs,f)
    print(model_dm.vector_size)
    print(len(model_dm.docvecs))
    print("Process finish")