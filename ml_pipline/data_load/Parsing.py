import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from dataclasses import dataclass
import logging
import click

SLEEP = 3
DEPTH = 100
URL = 'https://psi.mchs.gov.ru/vopros-psihologu'

logging.basicConfig(level=logging.INFO)
@dataclass
class Message:
    title: str = None
    content: str = None
    datetime: str = None
    pagen: int = None

# Настраиваем вебдрайвер
chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument('headless')
chrome_options.add_argument('no-sandbox')
driver = webdriver.Chrome(options=chrome_options)



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
def load_data(data_path:str) -> None:
    for page in range(201,205):
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

    df =pd.DataFrame(data=dataset)
    df.to_csv(data_path, index=False)
    logging.info("Data saved successfully.")
    driver.close()

if __name__ == "__main__":
    load_data()