FROM python:3.12-slim


# Создание рабочей директории
WORKDIR /workdir

# Копирование файлов
COPY requirements.txt TrainLDA.py /workdir/

# Установка зависимостей
RUN pip install -r requirements.txt