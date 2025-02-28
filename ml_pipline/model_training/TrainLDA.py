import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

N=10000
n_topics = 60
# Загрузка данных
df = pd.read_csv('normdata.csv')

# Проверка размера датасета

# Выбор столбца с текстами
df = df.dropna()
texts = df['content'].tolist()
#texts = texts[0:N]
print(f"Размер датасета: {len(texts)}")
# Скачивание русских стоп-слов
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
#with open('stopwords-ru.json', 'r', encoding='utf-8') as f:
#    stopwords_json = json.load(f)
with open('RussianStopWords.txt', 'r', encoding='utf-8') as f:
    stopwords_txt = [line.strip() for line in f if line.strip()]
russian_stopwords = list(set(russian_stopwords + stopwords_txt ))
russian_stopwords = [word.lower() for word in russian_stopwords]
russian_stopwords.extend(['хочу', 'могу','ключ', 'идти', 'делать', 'стало', 'недели', 'неделя','знаю' ])
stemmer = SnowballStemmer("russian")
# Загрузка русской модели spaCy
nlp = spacy.load("ru_core_news_sm")

# Предобработка текста



# Создание матрицы термин-документ
count_vect = CountVectorizer(input='content', stop_words=russian_stopwords, min_df=5, max_df=0.9)
dataset = count_vect.fit_transform(texts)
# Обучение модели LDA

lda = LDA(n_components=n_topics, max_iter=50, learning_method='batch', random_state=42)
lda.fit(dataset)
import joblib
joblib.dump(lda, f'lda_model_size2.5k_comp{n_topics}.pkl')

# Получение тематических распределений для документов
doc_topic_distr = lda.transform(dataset)

# Кластеризация с помощью K-Means
n_clusters = 3
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
clust_labels = spectral.fit_predict(doc_topic_distr)

# Ограничение данных для визуализации
doc_topic_distr_subset = doc_topic_distr[:10000]
clust_labels_subset = clust_labels[:10000]


# Функция для получения топ-тем для каждого кластера
def get_top_topics_for_clusters(doc_topic_distr, clust_labels, n_top_topics=3):
    cluster_topics = {}
    unique_labels = np.unique(clust_labels)

    for label in unique_labels:
        if label == -1:  # Пропускаем шумовые точки (если используете DBSCAN)
            continue

        # Выбираем документы, принадлежащие текущему кластеру
        cluster_docs = doc_topic_distr[clust_labels == label]

        # Вычисляем среднее распределение тем для кластера
        mean_topic_dist = cluster_docs.mean(axis=0)

        # Находим индексы топ-тем
        top_topic_indices = mean_topic_dist.argsort()[-n_top_topics:][::-1]

        # Сохраняем топ-темы для кластера
        cluster_topics[label] = top_topic_indices

    return cluster_topics


# Получение топ-тем для каждого кластера
cluster_topics = get_top_topics_for_clusters(doc_topic_distr, clust_labels, n_top_topics=3)


# Функция для вывода топ-слов для тем
def print_top_words_for_topics(lda_model, feature_names, topic_indices, n_top_words=10):
    for idx in topic_indices:
        topic = lda_model.components_[idx]
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Тема #{idx}: {' '.join(top_words)}")


# Вывод топ-тем для каждого кластера
tf_feature_names = count_vect.get_feature_names_out()

for cluster_id, topic_indices in cluster_topics.items():
    print(f"Кластер #{cluster_id}:")
    print_top_words_for_topics(lda, tf_feature_names, topic_indices, n_top_words=5)
    print("\n")



# Альтернатива: Визуализация с помощью UMAP
try:
    umap_embeddings = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine',
                           random_state=42).fit_transform(doc_topic_distr_subset)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=clust_labels_subset, cmap='viridis', s=10,
                          alpha=0.5)
    plt.colorbar(scatter, label="Cluster Labels")
    plt.title("UMAP Visualization of Clusters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()
except Exception as e:
    print(f"Ошибка при работе с UMAP: {e}")

# Визуализация с подписями тем
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=clust_labels_subset, cmap='viridis', s=10,
                      alpha=0.5)

# Добавление подписей для центров кластеров
centroids = spectral.cluster_centers_
for i, centroid in enumerate(centroids):
    if i not in cluster_topics:  # Пропускаем пустые кластеры
        continue

    # Получаем топ-тему для кластера
    top_topic_idx = cluster_topics[i][0]
    top_words = [tf_feature_names[j] for j in lda.components_[top_topic_idx].argsort()[:-6 - 1:-1]]

    # Добавляем текстовую метку
    plt.text(centroid[0], centroid[1], f"C{i}: {' '.join(top_words)}", fontsize=8)

plt.colorbar(scatter, label="Cluster Labels")
plt.title("t-SNE Visualization of Clusters with Topics")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()