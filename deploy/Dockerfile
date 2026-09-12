# ShieldTab backend -- deploy target: Render (free tier) or any Docker host
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ ./model/
COPY data/ ./data/
COPY backend/ ./backend/

# Render sets $PORT at runtime; main.py reads it (falls back to 7860 locally)
EXPOSE 7860

WORKDIR /app/backend
CMD ["python", "main.py"]
