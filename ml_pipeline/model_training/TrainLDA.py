import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import click
import joblib
import logging

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--n_topics', help='Количество тем LDA')
@click.option('--data_path', help='Путь файла с данными')
@click.option('--model_path', help='Путь, где сохранится модель')
def train(data_path:str, n_topics: int, model_path:str):
    logging.info('Подгружаем данные')
    df = pd.read_csv(data_path)
    print(len(df))
    df.dropna(inplace=True)
    print(len(df))
    texts = df['content'].tolist()
    logging.info('Подгрузили')

    # Создание матрицы термин-документ
    count_vect = CountVectorizer(input='content', min_df=5, max_df=0.9)
    dataset = count_vect.fit_transform(texts)

    # Обучение модели LDA
    logging.info('Начинаем обучать')
    lda = LDA(n_components=int(n_topics), max_iter=50, learning_method='batch', random_state=42)
    lda.fit(dataset)
    logging.info('Обучили')
    joblib.dump(lda, f'{model_path}.pkl')
    logging.info('Выгрузили')

if __name__ == "__main__":
    train()

