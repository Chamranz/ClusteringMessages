import pandas as pd
import nltk
from nltk.corpus import stopwords
import logging
import click
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

logging.basicConfig(level=logging.INFO)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
tagger = NewsMorphTagger(emb)

def prep_text(text) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(tagger)

    nltk.download('stopwords')
    russian_stopwords = stopwords.words("russian")
    russian_stopwords = [word.lower() for word in russian_stopwords]
    russian_stopwords.extend(['хочу', 'могу', 'ключ', 'идти', 'делать', 'стало', 'недели', 'неделя', 'знаю'])

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma)>2]
    filtered_words = [word for word in words if word not in russian_stopwords]
    return ' '.join(filtered_words)


@click.command()
@click.option("--data_path", help="Путь к файлу с данными")
@click.option("--cleardata_path", help="Путь, где сохраняем обработанные файлы")

def upload_text(data_path: str, cleardata_path: str) -> None:
    df = pd.read_csv(data_path)
    df = df.dropna()
    df['content'] = df.content.apply(prep_text)
    df.to_csv(cleardata_path)

if __name__ == "__main__":
    upload_text()



