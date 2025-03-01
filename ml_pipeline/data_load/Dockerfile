FROM python:3.12-alpine

# Обновление системы и установка зависимостей
RUN apk add --no-cache \
    bash \
    chromium \
    chromium-chromedriver \
    nss \
    freetype \
    ttf-dejavu \
    dbus-glib
# Установка bash как default shell

COPY requirements.txt /workdir/
WORKDIR /workdir

RUN pip install -r requirements.txt

COPY Parsing.py /workdir/
