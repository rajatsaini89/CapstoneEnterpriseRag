FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir --no-compile && \
    pip install --no-cache-dir --no-compile -r requirements.txt

COPY config.json .
COPY src ./src
COPY evaluation_questions.db .
COPY Docs /app/Docs

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]