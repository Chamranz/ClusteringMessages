import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dataclasses import dataclass
import logging
import click
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

SLEEP = 3
DEPTH = 100
URL = 'https://psi.mchs.gov.ru/vopros-psihologu'

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
tagger = NewsMorphTagger(emb)

logging.basicConfig(level=logging.INFO)

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

@dataclass
class Message:
    title: str = None
    content: str = None
    datetime: str = None
    pagen: int = None

# Настраиваем вебдрайвер
chrome_options = Options()
chrome_options.add_argument("--headless")  # Безголовый режим
chrome_options.add_argument("--no-sandbox")  # Отключение песочницы
chrome_options.add_argument("--disable-dev-shm-usage")  # Использование /tmp вместо /dev/shm
chrome_options.add_argument("--remote-debugging-port=9222")  # Порт для удаленной отладки
chrome_options.add_argument("--disable-gpu")  # Отключение GPU acceleration
chrome_options.add_argument("--window-size=1920,1080")  # Размер окна

# Увеличение таймаута для инициализации драйвера
chrome_options.page_load_strategy = 'normal'  # Стратегия загрузки страницы
driver = webdriver.Chrome(options=chrome_options)
# Увеличение таймаута до 300 секунд
driver.set_page_load_timeout(300)
driver.implicitly_wait(300)

# Пример использования WebDriverWait
try:
    WebDriverWait(driver, 300).until(
        EC.presence_of_element_located((By.CLASS_NAME, "public_appeals_list"))
    )
except Exception as e:
    logging.error(f"Timeout error: {e}")

def parse_page(page, pagen):
    driver.get(page)
    time.sleep(1)
    time.sleep(SLEEP)
    html = driver.page_source
    # варим суп страницы
    soup = BeautifulSoup(html, "html.parser")
    obj = soup.find('div', {'class': 'public_appeals_list'})

    messages = obj.find_all('div', {'class': 'request-card'})
    messages = messages[::2]
    data = []
    for item in messages:
        message = Message()

        message.pagen = pagen

        title = item.find('div', {'class': 'request-card__title'}).text
        message.title = title

        times = item.find('div', {'class': 'request-card__top-right'}).text
        message.datetime = times

        content = item.find('div', {'class': 'request-card__content'})
        mes_text = ''
        stop = False
        tag_num = 0
        try:
            content = content.find('a').text
            mes_text+=content
        except:
            content = content.find_all('p')
            while stop == False:
                text = str(content[tag_num])
                clean_text = re.sub(r"<p>(.*?)</p>", r"\1", text, flags=re.DOTALL)
                if tag_num!=0:
                    if clean_text=='':
                        stop = True
                        continue
                    else:
                        first_word = clean_text.split()[0]
                        if first_word == 'Здравствуйте' or first_word == 'Здравствуйте,' or first_word == 'Здравствуйте!':
                            stop = True
                            tag_num += 1
                            continue
                        else:
                            mes_text+=clean_text
                else:
                    if clean_text=='':
                        tag_num += 1
                        continue
                    else:
                        mes_text+=clean_text
                tag_num+=1
        message.content = mes_text.strip()
        data.append(message)

    return data

# Последовательно парсим все собранные страницы
data = []

@click.command()
@click.option('--data_path', help='Путь, где сохранятся наши данные')
@click.option('--page_number', help='Номер страницы, до которой парсим (обновляем) данные', type=int)
def load_data(data_path: str, page_number: int) -> None:
    for page in range(0,int(page_number)):
        try:
            pagen = page
            res = parse_page('https://psi.mchs.gov.ru/vopros-psihologu' + f'?page={page}', pagen)
            data.append(res)
            logging.info("News page fetched successfully.")

        except:
            logging.info("News fetched UNsuccessfully.")
            pass

    dataset = []
    for page in data:
        for mess in page:
            dataset.append(mess)

    df = pd.DataFrame(data=dataset)
    df = df.dropna()
    df['content'] = df.content.apply(prep_text)
    df.to_csv(data_path, index=False)
    logging.info("Data saved successfully.")
    driver.close()

if __name__ == "__main__":
    load_data()