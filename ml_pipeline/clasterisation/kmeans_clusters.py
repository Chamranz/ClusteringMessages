import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import click
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--n_clusters', help = 'Количество класторов')
@click.option('--P', help = 'Гиперпараметр TSNE')
@click.option('--model_path', help = 'Путь до модели LDA')
@click.option('--data_path', help = 'Путь до данных')
@click.option('--save_pic', help = 'Путь, где сохранится рисунок кластеризации')
def clustering(n_clusters:int, P:int, model_path: str, data_path: str,
               save_pic: str):

    # Загрузка данных
    df = pd.read_csv(data_path)
    data = df['content'].tolist()

    # count vectorizer
    count_vect = CountVectorizer(input='content')
    term_doc_matrix = count_vect.transform(data)
    lda = joblib.load(model_path)
    embeddings = lda.transform(term_doc_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clust_labels = kmeans.fit_predict(embeddings)
    clust_centers = kmeans.cluster_centers_

    embeddings_to_tsne = np.concatenate((embeddings,clust_centers), axis=0)
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
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=clust_labels, cmap='viridis', s=10, alpha=0.5)
    centroids = plt.scatter(centroids_embeddings[:, 0], centroids_embeddings[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.legend(handles=[scatter, centroids])
    plt.title(f"t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label="Cluster Labels")
    plt.savefig(save_pic)