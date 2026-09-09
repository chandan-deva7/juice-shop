FROM python:3.12-slim

WORKDIR /app
COPY ai_assistant/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_assistant ./ai_assistant

EXPOSE 8000
CMD ["uvicorn", "ai_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]