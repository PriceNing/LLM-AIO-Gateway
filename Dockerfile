FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY app/ app/
COPY docker-entrypoint.sh .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

VOLUME ["/app/data", "/app/logs"]

ENV LLM_GATEWAY_CONFIG=/app/data/config.json

ENTRYPOINT ["/app/docker-entrypoint.sh"]
