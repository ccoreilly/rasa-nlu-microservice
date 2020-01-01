FROM python:3.7.6 as base

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY cache.py .
COPY server.py .
COPY train.py .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "server:app"]

FROM base as catalan

RUN pip install https://github.com/assistent-cat/bot/releases/download/v0.0.1/ca_fasttext-0.0.1.tar.gz \
    && python -m spacy link ca_fasttext ca