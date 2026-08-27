FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Allow bare imports within src/ (e.g. "from business_rules import ...")
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]