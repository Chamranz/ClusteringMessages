FROM python:3.12-slim
# Установка компиляторов и зависимостей для сборки пакетов

COPY requirements.txt kmeans_clusters.py /workdir/

WORKDIR /workdir
# Установка зависимостей
RUN pip install -r requirements.txt
