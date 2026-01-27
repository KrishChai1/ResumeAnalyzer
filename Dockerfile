FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY api_server_minimal.py ./api_server.py

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT}
