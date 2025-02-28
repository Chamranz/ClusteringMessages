import re
from pymystem3 import Mystem
import numpy as np
import spacy
from nltk.corpus import stopwords
import nltk
import dask.dataframe as dd
from dask_ml.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import json
from sklearn.cluster import DBSCAN

#ДИМА, РАБОТАЕМ ТУТ!!
num_exp = 2
#n_comp = 6
n_clusters = 3
P = 30
N=10000
print(f'n_clusters = {n_clusters}, P = {P}, N = {N}', f"num _exp = {num_exp}")
#играмся с этими 4-я переменными



# Скачивание русских стоп-слов
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
#with open('stopwords-ru.json', 'r', encoding='utf-8') as f:
#    stopwords_json = json.load(f)
with open('RussianStopWords.txt', 'r', encoding='utf-8') as f:
    stopwords_txt = [line.strip() for line in f if line.strip()]
russian_stopwords = list(set(russian_stopwords + stopwords_txt ))
russian_stopwords = [word.lower() for word in russian_stopwords]
russian_stopwords.extend(['хочу', 'могу','ключ' ])
# Применение функции предобработки к столбцу 'content'
import pandas as pd

# Загрузка данных
df = pd.read_csv('smaller_data.csv')
df = df.iloc[0:N]
print('прочитали объем')
train_names = df['content'].tolist()
print('выделили нужный объем')

# count vectorizer
count_vect = CountVectorizer(input='content',
                             stop_words=russian_stopwords
                            )
dataset = count_vect.fit_transform(train_names)
print(f'shape = {dataset.shape}')

#tf_idf

#tf_idf = TfidfVectorizer(input='content', stop_words=russian_stopwords, smooth_idf=False)
#tf_idf.fit(train_names)
#print(f'len = {len(tf_idf.get_feature_names_out())}')
#idfs = tf_idf.idf_
#lower = 3
#upper = 6
#not_often = idfs > lower
#not_rare = idfs < upper

#mask = not_often * not_rare
#
#good_words = np.array(tf_idf.get_feature_names_out())[mask]
#cleaned = []
#for word in good_words:
#    word = re.sub(r"^\d+\w*$|_+", "", word)

#    if len(word) == 0:
#        continue
#    cleaned.append(word)
#print(len(cleaned))


term_doc_matrix = count_vect.transform(train_names)
import joblib
lda = joblib.load('lda_model_comp3.pkl')
embeddings = lda.transform(term_doc_matrix)

#joblib.dump(lda, f'lda_model_size50k_comp{n_comp}.pkl')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clust_labels = kmeans.fit_predict(embeddings)
clust_centers = kmeans.cluster_centers_

embeddings_to_tsne = np.concatenate((embeddings,clust_centers), axis=0)
from sklearn.manifold import TSNE
tSNE =  TSNE(n_components=2, perplexity=P, random_state=42)
tsne_embeddings = tSNE.fit_transform(embeddings_to_tsne)
tsne_embeddings, centroids_embeddings = np.split(tsne_embeddings, [len(clust_labels)], axis=0)

n_top_words = 10

# Получение словаря (vocabulary) из CountVectorizer
tf_feature_names = count_vect.get_feature_names_out()

# Вывод топ-слов для каждой темы
for topic_idx, topic in enumerate(lda.components_):
    top_words_indices = topic.argsort()[:-n_top_words - 1:-1]  # Индексы топ-слов
    top_words = [tf_feature_names[i] for i in top_words_indices]  # Слова по индексам
    print(f"Тема #{topic_idx}: {' '.join(top_words)}")

# Визуализация
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=clust_labels, cmap='viridis', s=10, alpha=0.5)
centroids = plt.scatter(centroids_embeddings[:, 0], centroids_embeddings[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.legend(handles=[scatter, centroids])
plt.title(f"t-SNE Visualization of Clusters, N = {num_exp}")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter, label="Cluster Labels")
plt.show()