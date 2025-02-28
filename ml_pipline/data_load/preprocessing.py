import pandas as pd
import nltk
import preprocessing_tools

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
import logging
import click

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--data_path", help="Путь к файлу с данными")
@click.option("--cleardata_path", help="Путь, где сохраняем обработанные файлы")
def preprocces(data_path: str, cleardata_path: str) -> None:
    df = pd.read_csv(data_path)
    df = df.dropna()
    nltk.download('stopwords')
    russian_stopwords = stopwords.words("russian")
    russian_stopwords = [word.lower() for word in russian_stopwords]
    russian_stopwords.extend(['хочу', 'могу','ключ', 'идти', 'делать', 'стало', 'недели', 'неделя','знаю' ])
